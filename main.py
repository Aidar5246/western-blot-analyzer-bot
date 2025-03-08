from config import token
from aiogram import Bot, Dispatcher, F, types
import asyncio
from aiogram.filters.command import Command
from datetime import datetime
from function import *
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
import openpyxl

bot = Bot(token=token)
dp = Dispatcher()

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привет, я твой новый персональный научный помощник. Приступим к работе?")

# Хэндлер на команду /help
@dp.message(Command("help"))
async def cmd_start(message: Message):
    await message.answer("1.Отправьте изображение иммуноблотинга\n\
                        2. Выберите интересеующую часть изображения\n\
                        3. Получите готовый результат анализа")

# Обработка фото
@dp.message(F.photo)
async def send_answer(message: Message):

    await message.answer('⌛️')
    previous_answer_id = message.message_id + 1

    # скачиваю отправленные боту изображения в архив
    file_name = f"photos/{datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")}__{message.from_user.id}.jpg"
    await bot.download(message.photo[-1], destination=file_name)

    # вызываю функцию предсказания
    photo, num_clusters = await model_predict(file_name, message.from_user.id)
    
    if num_clusters == 0:

        await bot.delete_message(message_id = previous_answer_id, chat_id = message.chat.id)
        await message.answer("По вашему изображению нет иммуноблотинга")
        return

    # создаю инлайн кнопки
    builder = InlineKeyboardBuilder()
    for i in range(num_clusters):
        builder.button(text = f'Образец {i+1}', callback_data=f"{i+1}")
    builder.adjust(4)


    await bot.delete_message(message_id = previous_answer_id, chat_id = message.chat.id)
    # отправляю результат 
    await message.answer_photo(photo=types.FSInputFile(path=photo))
    await message.answer("Пожалуйста, выберите интересующую вас область изображения", reply_markup = builder.as_markup()) 


# Обработка инлайн кнопок
@dp.callback_query()
async def process_any_inline_button_press(callback: CallbackQuery):
    await callback.message.answer_photo\
                    (photo = types.FSInputFile(path = f'results/user_{callback.message.chat.id}/img_{callback.data}.png'))
            
    await callback.message.answer_document\
                    (document = types.FSInputFile(path = f'results/user_{callback.message.chat.id}/result_blot_{callback.data}.xlsx'))

# Обработка текста
@dp.message(F.text)
async def send_answer(message: Message):
    await message.answer('Пожалуйста, отправьте изображение иммуноблотинга')
                        

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
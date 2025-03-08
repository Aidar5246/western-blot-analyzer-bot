from ultralytics import YOLO
from PIL import Image, ImageStat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import os
import shutil


# Загрузка обученной модели
model = YOLO('model/best.pt')  


# Предобработка изображения
async def preprocess_img(path):
    img = Image.open(path)
    img = img.convert("L")
    return img


async def model_predict(path, user_id):
    try: shutil.rmtree(f'results/user_{user_id}')
    except: pass

# создаем отдельную папку для нашего user
    try: os.mkdir(f'results/user_{user_id}')
    except FileExistsError: pass
    
    img_sample = await preprocess_img(path)
    res = model.predict(img_sample)
    list_data = []

    # Получение координат и вероятностей
    for box in res[0].boxes:
        list_data.append(box.data[0].squeeze(0).tolist())
    df = pd.DataFrame(list_data, columns=['x1', 'y1', 'x2', 'y2', 'probability', 'class']).drop(columns=['class'])

    if df.empty: return 0, 0

    # Get intensive
    df['intensive'] = df.apply(lambda x: brightness(img_sample.crop((x['x1'], x['y1'], x['x2'], x['y2']))), axis=1)

    # По оси y сортирую от меньшего к большому, далее срнавниваю разброс от центров соседних блотов (отклонение не должно быть больше 5%)
    # в таком случае подразумевается что блоты находяться на одной горизонтальной линии 

    df_sort_y1 = df.sort_values(by=['y1'], ascending=True, ignore_index=True)
    a = 1
    count_dict = {}
    count_dict[a] = [str(df_sort_y1['y1'][0])]
    for i in range(len(df_sort_y1) - 1):

        # hight = df_sort_y1['y2'][i] - df_sort_y1['y1'][i]
        # median_current_y = (df_sort_y1['y2'][i] + df_sort_y1['y1'][i])/2
        median_next_y = (df_sort_y1['y2'][i+1] + df_sort_y1['y1'][i+1])/2

        next_y = df_sort_y1['y1'][i+1]

        if df_sort_y1['y1'][i] < median_next_y < df_sort_y1['y2'][i]:
            count_dict[a].append(str(next_y))

        else:
            a += 1
            count_dict[a] = [str(df_sort_y1['y1'][i+1])]

    # Вывожу количество линий по белковым массам
    for group in count_dict.keys():
        for coord in count_dict[group]:
            df.loc[df['y1'] == float(coord), 'group_id'] = group
    df_group = df.sort_values(by = ['group_id','x1'], ignore_index = True, ascending=True)
    

    x = np.array(Image.open(path), dtype=np.uint8) 
    # Create figure and axes (только ради того чтобы получить координаты)
    fig, ax = plt.subplots(1)
    ax.imshow(x) 
    plt.close(fig)
    ymin,ymax = ax.get_ylim()
    xmin,xmax = ax.get_xlim()

    # создаем датасет средних показателей для линии групп
    xz = df_group.groupby('group_id').mean().reset_index()

    xz['diff'] = (xz['y1'] - xz['y2'].shift())/2

    # если верхняя пустая часть до блота будет больше чем вмещается 4 высоты блота, то возьму 30% длину в 3 блота
    if (xz.loc[0, 'y1']/2) // (xz.loc[0, 'y2'] - xz.loc[0, 'y1']) > 4:
        xz.loc[0, 'diff'] = (xz.loc[0, 'y2'] - xz.loc[0, 'y1']) * 3

    # если вмещается меньше, то возьму только верхнюю половину до блота
    else:
        xz.loc[0, 'diff'] = xz.loc[0, 'y1']/2
        
    xz['y_high'] = (xz['y1'] - xz['diff'])
    xz['y_low'] = xz['y2'] + xz['diff'].shift(-1)
    # узнаем координаты конца y
    last_row = len(xz)-1

    # если нижняя пустая часть от блота будет больше чем вмещается 4 высоты блота, то возьму 30% длину в 3 блота
    if (ymin - xz.loc[last_row, 'y2']) // (xz.loc[last_row, 'y2'] - xz.loc[last_row, 'y1']) > 4:
        xz.loc[last_row, 'y_low'] = (xz.loc[last_row, 'y2'] - xz.loc[last_row, 'y1']) * 3 + xz.loc[last_row, 'y2']
    # если вмещается меньше, то возьму только нижнюю половину до блота
    else:
        xz.loc[last_row, 'y_low'] = (ymin - xz.loc[last_row, 'y2'])/2 + xz.loc[last_row, 'y2']


    # вырезаем область с линией блотов, чтобы пользователь выбрал интересующую его область
    im = Image.open(path)
    list_img = []
    for i in range(len(xz['group_id'])):
        im_crop = im.crop((xmin, xz['y_high'].iat[i], xmax, xz['y_low'].iat[i]))
        list_img.append(im_crop)

    len_img = len(list_img)
    fig, axes = plt.subplots(len_img, 1)

    if len_img == 1:
        axes.imshow(list_img[0])
        axes.set_title(f'n_cluster = 1')
        axes.axis('off')
    else:
        for i in range(len_img):
            axes[i].imshow(list_img[i])
            axes[i].set_title(f'n_cluster = {i+1}')
            axes[i].axis('off')

    plt.savefig(f'results/user_{user_id}/clustering_img.png')
    plt.close(fig)
 

# теперь создадим для каждой молекулярной массы свой собственный датасет
    for name_fig in range(len_img):
        name_fig = name_fig + 1
        # выбираем необходимые датасеты
        box_blot = df_group[df_group['group_id'] == name_fig].reset_index()
        fig_blot = xz[xz['group_id'] == name_fig].reset_index()


        x = np.array(Image.open(path), dtype=np.uint8) 
        # plt.imshow(x) 
        
        # Create figure and axes 
        fig, ax = plt.subplots(1) 
        
        # Display the image 
        ax.imshow(x) 
        
        for coordinate in range(len(box_blot)):
            # Create a Rectangle patch 
            rect = patches.Rectangle(xy = [box_blot['x1'].iat[coordinate], box_blot['y1'].iat[coordinate]], 
                                    width = box_blot['x2'].iat[coordinate]- box_blot['x1'].iat[coordinate], 
                                    height = box_blot['y2'].iat[coordinate]- box_blot['y1'].iat[coordinate],  
                                    linewidth=1, 
                                    edgecolor='r', 
                                    facecolor="none") 
            
            # Add the patch to the Axes 
            ax.add_patch(rect) 

        xmin,xmax = ax.get_xlim()

        plt.xlim([xmin,xmax]) 
        plt.ylim([fig_blot['y_low'][0], fig_blot['y_high'][0]]) 
        plt.axis('off')
        # %matplotlib inline
        plt.savefig(f'results/user_{user_id}/img_{name_fig}.png')
        plt.close(fig)

        # если линий белков 1
        count_dict_x = {}
        count_dict_x['1'] = [str(box_blot['intensive'][0])]
        lenn_ = len(box_blot)

        if lenn_ == 0:
            final_df = pd.DataFrame([0], index = ['Интенсивность'], columns = [1])

        elif lenn_ == 1:                 
            final_df = pd.DataFrame(count_dict_x['1']).T
            final_df.columns = ['1']
            final_df = final_df.rename(index={0:'Интенсивность'})

        elif lenn_ > 1:
            for i in range(lenn_ - 1):
                current_x = box_blot['x2'][i]
                next_x = box_blot['x1'][i+1]
                lenght_blot = box_blot['x2'][i] - box_blot['x1'][i]
                distance = next_x - current_x

                # проверка на нормальное распределение
                if distance < (lenght_blot * 0.7):
                    count_dict_x['1'].append(str(box_blot['intensive'][i+1]))
                
                # в случае если есть пропуск
                else:          
                    coef_ = distance//(lenght_blot * 0.9)
                    for r in range(int(coef_)):
                            count_dict_x['1'].append(0)
                    
                    count_dict_x['1'].append(str(box_blot['intensive'][i+1]))
                
            final_df = pd.DataFrame(count_dict_x['1']).T
            final_df.columns = range(1, len(final_df.columns)+1)
            final_df = final_df.rename(index={0:'Интенсивность'})


        # Min-Max нормализация по строкам
        final_df = final_df.apply(pd.to_numeric, errors='coerce')
        min_vals = final_df.min(axis=1).to_numpy().reshape(-1, 1)
        max_vals = final_df.max(axis=1).to_numpy().reshape(-1, 1)
        final_df = round(1 - (final_df - min_vals) / (max_vals - min_vals), 3)

        # сохранение полученных результатов
        final_df.to_excel(f'results/user_{user_id}/result_blot_{name_fig}.xlsx', index = False)
        path_to_cluster_img = f'results/user_{user_id}/clustering_img.png'
    
    return path_to_cluster_img, name_fig
        

# Определние интенсинвости белка
def brightness(im_file):
    stat = ImageStat.Stat(im_file)
    return stat.mean[0]
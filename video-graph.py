import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from math import sqrt

model = keras.models.load_model('result')

# Количество сегментов на который делится видеоролик
# Подобранно эксперементально. ВНИМАНИЕ: Будет менять амплитуду графика
# Значения ниже уменьшают чувствительность сети к ДТП
# Значения выше соответственно увеличивают
squares = 13

accident_probabilities = []
deriative = []

truth = []

for x in open("test/truth.txt").readlines():
    truth.append(x[0] == "1")
print(truth)

def plot(video: str = "2.mp4"):
    cap = cv2.VideoCapture(f"./{video}")
    i = 0
    p = -1
    ret = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while ret:
        ret, frame = cap.read()
        i += 1
        if (i > fps):
            p += 1
            i = 0
        else:
            continue

        im = Image.fromarray(frame)

        accident_probability = 0
        images_noshift = []
        images_withshift = []
        ims = im.size[0] / squares
        
        shift_x = min(im.size[0] - int(im.size[0] / ims) * ims, 64)
        shift_y = min(im.size[1] - int(im.size[1] / ims) * ims, 64)
        
        for x in range(0, int(im.size[0] / ims)):
            for y in range(0, int(im.size[1] / ims)):
                img_array = keras.preprocessing.image.img_to_array(im.crop((x * ims, y * ims, x * ims + ims, y * ims + ims)).resize((96, 96), 2))
                img_array = tf.expand_dims(img_array, 0)
                images_noshift.append(img_array)

                img_array = keras.preprocessing.image.img_to_array(im.crop((x * ims, y * ims, x * ims + ims + shift_x, y * ims + ims + shift_y)).resize((96, 96), 2))
                img_array = tf.expand_dims(img_array, 0)
                images_withshift.append(img_array)
    
        predictions = model.predict(np.vstack(images_noshift), batch_size=128)
        predictions2 = model.predict(np.vstack(images_withshift), batch_size=128)

        z = 0
        for score in predictions:
            score2 = predictions2[z]
            if (score[1] > 0.2) and (score2[1] > 0.2):
                # Мы используем среднее между 2мя попытками для снижения количества ложных сигналов
                accident_probability += (score[1] + score2[1]) / 2
            z += 1
        accident_probabilities.append(accident_probability)
        if p > 0:
            # Мы используем производную графика общего рейтинга аварии, это позволяет нам легко
            # находить изменения
            deriative.append(max(accident_probability - accident_probabilities[p-1], 0))

d = 0
fails = 0
dirs = listdir("./test/")
for x in dirs:
    if not x.endswith(".mp4"):
        continue
        
    print(x)
    
    deriative = []
    accident_probabilities = []
    
    plot("./test/"+x)
    
    i = 0
    has_peak = False
    dm = 0
    
    print("Peaks:")
    for k in deriative:
        if (k < 0.5):
            deriative[i] = 0
        # При наличии скачка в производной графика, мы считаем что на видеоролике присутствует авария
        # Число 0.9 подобранно эксперементальным путем.
        # Оно может иметь смысл, так как при аварии обычно сеть выдает значения выше 0.9
        # А при низкой вероятности нам могут попастся 2 квадрата по 0.45
        elif (k > 0.9):
            has_peak = True
            print("** peak at: "+str((i*30)/60))
        if k > dm:
            dm = k
        i += 1
    print("*Max value: "+str(round(dm, 2))+"*")
    
    t = int(x[0] + x[1].replace('.', '')) - 1
    print("|t: "+str(truth[t])+" vs p: "+str(has_peak)+"|")
    if truth[t] != has_peak:
        fails += 1
        
    print("-> Progress: "+(str(round(d/len(dirs), 2)))+". Score: "+str(round((1-fails/(d + 1)), 2)))
    d += 1

print("Result: "+str((1-fails/d) * 100)+"%")

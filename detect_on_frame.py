import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


model = keras.models.load_model('result')

im = Image.open('test.png').convert("RGB")

fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

ims = 146


for x in range(0, int(im.size[0] / ims)):
    for y in range(0, int(im.size[1] / ims)):

        img_array = keras.preprocessing.image.img_to_array(im.crop((x * ims, y * ims, x * ims + ims, y * ims + ims)).resize((96, 96), 2))
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]

        rect = patches.Rectangle((ims* x, ims * y), ims, ims, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        maxscore = score.argmax()
        a = score[maxscore] / 1.5
        if (maxscore == 1):
            rect = patches.Rectangle((ims * x, ims * y), ims, ims, linewidth=1, edgecolor='b', facecolor='r', alpha=a)
            ax.add_patch(rect)
            
        if (maxscore == 0):
            rect = patches.Rectangle((ims * x, ims * y), ims, ims, linewidth=1, edgecolor='b', facecolor='yellow', alpha=a)
            ax.add_patch(rect)

        if (maxscore == 2):
            rect = patches.Rectangle((ims * x, ims * y), ims, ims, linewidth=1, edgecolor='b', facecolor='g', alpha=a)
            ax.add_patch(rect)
        ax.text(ims * x + 15, ims * y + 32, maxscore, color="r")
        ax.text(ims * x + 15, ims * y + 64, round(score[maxscore], 2), color="r")

plt.show()

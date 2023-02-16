# James William Fletcher (github.com/mrbid)
import sys
import numpy as np
from tensorflow import keras

np.set_printoptions(threshold=sys.maxsize)

def normaliseImage(arr):
    arr = arr.flatten()
    newarr = []
    for x in arr:
        if x != 0:
            newarr.append(x/255)
        else:
            newarr.append(0)
    return newarr

img = keras.preprocessing.image.load_img("ud.png")
arr = keras.preprocessing.image.img_to_array(img)
data = normaliseImage(np.array(arr))
input = np.reshape(data, [-1, 28,28,3])
print(input)
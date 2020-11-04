import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = []
target = []
bgSubThreshold = 100

# data_path = "/Volumes/Samsung_T5/Data_Sets/isl_letters_dataset/ISL_Dataset"
# data_path = "C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\dataset\\ISL_Dataset"
data_path = "C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\dataset\\ISL DATASET - NEW"

# categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
#               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
#               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']

# labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# label_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12,
#             'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
#            'y': 24, 'z': 25, '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34,
#             '9': 35}
label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12,
              'd': 13, 'e': 14, 'f': 15, 'g': 16, 'i': 17, 'k': 18, 'l': 19, 'm': 20, 'n': 21, 'o': 22, 'p': 23,
              'q': 24, 'r': 25, 's': 26, 't': 27, 'u': 28, 'w': 29, 'x': 30, 'y': 31, 'z': 32}


def get_image_path(folder_path):
    path = []
    img = os.listdir(folder_path)
    for i in img:
        img_loc = i.split("._")
        if len(img_loc) > 1:
            pass
        else:
            path.append(img_loc[len(img_loc) - 1])
    return path


# 1. reading images
# 2. converting to grayscale
# 3. resizing to (100px) x (100px)
# 4. adding to data

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_folder = get_image_path(folder_path)

    for image in img_folder:
        img_path = os.path.join(folder_path, image)
        print(img_path)
        img = cv2.imread(img_path)
        # bgModel = cv2.BackgroundSubtractorMOG2(0, bgSubThreshold)
        # fgmask = bgModel.apply(img)
        # res = cv2.bitwise_and(img, img, mask=fgmask)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (100, 100))
        data.append(img_resized)
        target.append(label_dict[category])

data = np.array(data)
target = np.array(target)

data_final = np.reshape(data, (data.shape[0], 100, 100, 1))
targets_final = to_categorical(target)

print(data_final.shape)
print(targets_final.shape)


def get_train_test_split():
    train_data, test_data, train_target, test_target = train_test_split(data_final, targets_final, test_size=0.2)
    return train_data, test_data, train_target, test_target

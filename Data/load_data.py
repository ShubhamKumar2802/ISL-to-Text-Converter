import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = []
target = []
bgSubThreshold = 100

# data_path = "/Volumes/Samsung_T5/Data_Sets/isl_letters_dataset/ISL_Dataset"
data_path = "C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\dataset\\self made dataset"

categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
              'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22,
              'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}

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

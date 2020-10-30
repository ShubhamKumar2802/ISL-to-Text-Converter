from Data.data import data_final, targets_final

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data_final, targets_final, test_size=0.2)
print(train_data.shape)

model = Sequential()

model.add(Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.3))

model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_target, epochs=10, validation_split=0.2)
# model.save("/Volumes/Samsung_T5/Data_Sets/isl_model_cnn/isl")
model.save("C:\\Users\\Aniket\\Desktop\\MINI PROJECT")

# print(data)
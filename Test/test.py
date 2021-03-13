from Data.load_data import get_train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
import numpy as np

train_data, test_data, train_target, test_target = get_train_test_split()

model = keras.models.load_model("C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\our_dataset_new_model")
predictions = model.predict(test_data)
predictions_classes = np.argmax(predictions, axis=1)
test_target_classes = np.argmax(test_target, axis=1)

score = model.evaluate(test_data, test_target)
print(score)

acc = accuracy_score(test_target_classes, predictions_classes)
print(acc)

report = classification_report(test_target_classes, predictions_classes)
print(report)

cm = confusion_matrix(test_target_classes, predictions_classes)
print(cm)

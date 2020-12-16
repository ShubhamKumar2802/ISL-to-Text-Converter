from Data.data import get_train_test_split
import keras
from sklearn.metrics import classification_report
import keras

train_data, test_data, train_target, test_target = get_train_test_split()
model = keras.models.load_model("C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\our_dataset_model")
predictions = model.predict(test_data)
score = model.evaluate(test_data, test_target)
print(score)
report = classification_report(test_target, predictions.round())
print(report)

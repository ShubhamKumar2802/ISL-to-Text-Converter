from Data.data import get_train_test_split
import keras
from sklearn.metrics import classification_report

train_data, test_data, train_target, test_target = get_train_test_split()
model = keras.models.load_model("C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\newmodel3")
predictions = model.predict(test_data)
score = model.evaluate(test_data, test_target)
print(score)
report = classification_report(test_target, predictions.round())
print(report)

# for i in range(len(test_data)):
#   print(f'index = {i}')
#   print(f'test_target data: {np.argmax(test_target[i])}')
#   print(f'Predicted value: {np.argmax(predictions[i])}')
#   print('--------------------------------')
#   print()

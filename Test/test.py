from Data.load_data import get_train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data, test_data, train_target, test_target = get_train_test_split()

model = keras.models.load_model("C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\our_dataset_new_model")
predictions = model.predict(test_data)
predictions_classes = np.argmax(predictions, axis=1)
test_target_classes = np.argmax(test_target, axis=1)

score = model.evaluate(test_data, test_target)
print(score)

acc = accuracy_score(test_target_classes, predictions_classes)
print("Accuracy Score: ", acc)
corr_labels = accuracy_score(test_target_classes, predictions_classes, normalize=False)
print("Number of correctly predicted samples: ", corr_labels)

report = classification_report(test_target_classes, predictions_classes, digits=4)
print(report)

cm = confusion_matrix(test_target_classes, predictions_classes)
print(cm)

sns.set(font_scale=1.2)
fig = plt.figure(figsize=(40, 40), dpi=80)
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt="d", linewidths=0.2)
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
plt.show()

plt.figure(figsize=(12, 8))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(test_data[i])
    plt.xlabel(f"Actual: {test_target_classes[i]}\n Predicted: {predictions_classes[i]}")

plt.tight_layout()
plt.show()

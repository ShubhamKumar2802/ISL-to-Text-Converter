import matplotlib.pyplot as plt
from Models.new_model import history

print(history.history.keys())

fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot Model Accuracy vs Epochs
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['train_acc', 'val_acc'], loc='upper left')

# Plot Model Loss vs Epochs
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['train_loss', 'val_loss'], loc='upper left')

plt.show()

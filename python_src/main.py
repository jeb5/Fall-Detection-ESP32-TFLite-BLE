import tensorflow as tf
import argparse
import numpy as np
import data_preparation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers

tf.get_logger().setLevel('ERROR')

WINDOW_SIZE = 64
NUM_CHANNELS = 8
# channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, angle_change, custom_feature
EPOCHS = 600


# Fall classification model
# Must be small enough to run on microcontroller
# Essential to avoid overfitting due to small dataset size
# Architecture 1: Deeper CNN with Dropout
model = tf.keras.models.Sequential([
    layers.Input(shape=(WINDOW_SIZE, NUM_CHANNELS)),
    layers.Conv1D(16, 5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    layers.Conv1D(16, 3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

checkpoint_cb = ModelCheckpoint(
    filepath='models/new_model.keras',
    monitor='val_f1_score',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=0,
)

np.set_printoptions(precision=4, suppress=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

f1_score_metric = tf.keras.metrics.F1Score(name='f1_score', average='weighted')

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', f1_score_metric])


def main(args):
  # print(model.summary())
  train_dataset, val_dataset, test_dataset = data_preparation.load_data()
  # Get training y labels for class weight computation
  y = np.array([])
  for _, labels in train_dataset:
    y = np.concatenate((y, labels.numpy().flatten()))

  # classes = np.unique(y)
  # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
  # class_weight_dict = {cls.astype(int): weight.astype(float) for cls, weight in zip(classes, class_weights)}

  if not args.load_model:
    # history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_cb], class_weight=class_weight_dict)
    print("Training model...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_cb], verbose=0)
    print("Training complete.")
    print("Loading best model from checkpoint...")
    # plot_history(history)
  else:
    print("Loading model from disk...")
  best_model = tf.keras.models.load_model('models/new_model.keras')

  confuse(best_model, val_dataset, "Validation Dataset")
  confuse(best_model, test_dataset, "Test Dataset")

  # print("Evaluating on test dataset...")
  # best_model.evaluate(test_dataset, verbose=0)

  # converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
  # tflite_model = converter.convert()

  # with open("models/new_model.tflite", "wb") as f:
  #   f.write(tflite_model)


def confuse(model, dataset, title, threshold=0.5):
  y_true = np.array([])
  y_pred = np.array([])

  for x, y in dataset:
    preds = model.predict(x, verbose=0)
    y_true = np.concatenate((y_true, y.numpy().flatten()))
    y_pred = np.concatenate((y_pred, (preds > threshold).astype(int).flatten()))

  # classification report
  print(f"Classification Report for {title}:")
  print(classification_report(y_true, y_pred, digits=4))

  # cm = confusion_matrix(y_true, y_pred)
  # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
  # disp.plot(cmap=plt.cm.Blues)
  # plt.title(title)
  # plt.show()


def plot_history(history):

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  f1 = history.history['f1_score']
  val_f1 = history.history['val_f1_score']
  epochs = range(1, len(acc) + 1)

  plt.figure(figsize=(18, 5))

  plt.subplot(1, 3, 1)
  plt.plot(epochs, acc, 'b', label='Training acc')
  plt.plot(epochs, val_acc, 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.subplot(1, 3, 2)
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 3, 3)
  plt.plot(epochs, f1, 'b', label='Training F1 Score')
  plt.plot(epochs, val_f1, 'r', label='Validation F1 Score')
  plt.title('Training and validation F1 Score')
  plt.xlabel('Epochs')
  plt.ylabel('F1 Score')
  plt.legend()

  plt.tight_layout()

  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train and evaluate a fall model.")
  parser.add_argument('--load_model', action='store_true', help="Load a pre-trained model from disk.")
  args = parser.parse_args()
  main(args)

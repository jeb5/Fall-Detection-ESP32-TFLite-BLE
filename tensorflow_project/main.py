import tensorflow as tf
import numpy as np
import data_preparation
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

WINDOW_SIZE = 64
BATCH_SIZE = 32
NUM_CHANNELS = 6

np.set_printoptions(precision=4, suppress=True)

train_dataset, val_dataset, test_dataset = data_preparation.load_data()

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(WINDOW_SIZE, NUM_CHANNELS)),
  tf.keras.layers.Conv1D(16, 3, activation='relu'),
  tf.keras.layers.Conv1D(32, 5, activation='relu'),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

checkpoint_cb = ModelCheckpoint(
    filepath='best_model.keras',   # or 'checkpoints/best_model.keras'
    monitor='val_loss',            # metric to watch
    save_best_only=True,           # only keep best model
    save_weights_only=False,       # save full model (recommended)
    mode='min',                    # because lower val_loss is better
    verbose=1
)
early_stop_cb = EarlyStopping(
    monitor='val_loss',
    patience=15,          # wait for 15 epochs of no improvement
    restore_best_weights=True
)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

def main():


  history = model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[checkpoint_cb, early_stop_cb])
  print("Training complete.")
  print("Loading best model from checkpoint...")
  best_model = tf.keras.models.load_model('best_model.keras')

  plot_history(history)

  print("Evaluating on test dataset...")
  best_model.evaluate(test_dataset)

  # converter = tf.lite.TFLiteConverter.from_keras_model(model)
  # tflite_model = converter.convert()

  # # Save the converted model to a .tflite file
  # with open("model.tflite", "wb") as f:
  #     f.write(tflite_model)

def plot_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
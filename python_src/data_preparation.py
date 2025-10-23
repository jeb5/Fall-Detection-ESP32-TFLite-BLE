import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from naive import custom_feature


DATA_CSV = "data_sorted.csv"
FALL_CSV = "fall_video3_marks.csv"
CACHE_PATH = ".data_cache"

DATA_FILES = [["data/session1_data_sorted.csv", "data/session1_marks.csv"],
              ["data/session2_data_sorted.csv", "data/session2_marks.csv"],
              ["data/session3_data_sorted.csv", "data/session3_marks.csv"],
              ["data/session4_data_sorted.csv", "data/session4_marks.csv"],
              ["data/session5_data_sorted.csv", "data/session5_marks.csv"],
              ["data/session6_data_sorted.csv", "data/session6_marks.csv"],
              ["data/session8_data_sorted.csv", "data/session8_marks.csv"],
              ]

FALL_WINDOW_MS = 3000
N_SAMPLES_EACH_FALL_TRAIN = 4
WINDOW_SIZE = 96
BATCH_SIZE = 32

np.set_printoptions(precision=4, suppress=True)

def get_naive_classified_data(params):
  dog_sigma, acc_lp_window, angle_lp_window, angle_exp, threshold = params
  all_true_positives = []
  all_false_positives = []
  total_falls = 0
  total_real_falls_removed = 0
  for i in range(len(DATA_FILES)):
      data_file, fall_file = DATA_FILES[i]
      # Read fall file. 1 column, skip header
      falls_ms = np.loadtxt(fall_file, delimiter=",", skiprows=1, usecols=(0))  * 1000  # convert to ms
      # Read data file. 7 columns, skip header (time,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
      # Load time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z (cols 0,2,3,4,5,6,7)
      data = np.loadtxt(data_file, delimiter=",", skiprows=1, usecols=(0,2,3,4,5,6,7))
      times = data[:,0].copy()  # shape (N,)
      deltaTimes = np.diff(times, prepend=0)  # shape (N,)
      acc_data = data[:,1:4].T  # shape (3, N)
      custom_feature_data = custom_feature(acc_data, dog_sigma, acc_lp_window, angle_lp_window, angle_exp, use_doa=False)
      # Find the middle of each block, where custom_feature_data > threshold
      blocks = np.where(np.diff(np.concatenate(([0], custom_feature_data > threshold, [0]))))[0].reshape(-1, 2)
      # Get time of maximum in each block
      detected_falls = []
      for start, end in blocks:
          if end > start:
              max_index = np.argmax(custom_feature_data[start:end]) + start
              detected_falls.append(max_index)
      
      actual_fall_times = falls_ms.tolist()

      # For each actual/detected fall, if there is a deltaTime >300ms within +-60 indicies of it, remove it
      # If the fall is within WINDOW_SIZE/2 of the start or end of the data, also remove it
      suspicious_window = 60
      suspicious_delta_time_threshold = 300  # ms
      actual_falls_original = actual_fall_times.copy()
      for fall_time in actual_falls_original:
          fall_index = np.searchsorted(times, fall_time)
          start_index = max(0, fall_index - suspicious_window)
          end_index = min(len(deltaTimes), fall_index + suspicious_window)
          if np.any(deltaTimes[start_index:end_index] > suspicious_delta_time_threshold):
              actual_fall_times.remove(fall_time)
          elif fall_index < WINDOW_SIZE//2 or fall_index > len(data) - WINDOW_SIZE//2:
              actual_fall_times.remove(fall_time)
      total_real_falls_removed += len(actual_falls_original) - len(actual_fall_times)
      detected_falls_original = detected_falls.copy()
      for detected_fall_index in detected_falls_original:
          start_index = max(0, detected_fall_index - suspicious_window)
          end_index = min(len(deltaTimes), detected_fall_index + suspicious_window)
          if np.any(deltaTimes[start_index:end_index] > suspicious_delta_time_threshold):
              detected_falls.remove(detected_fall_index)
          elif detected_fall_index < WINDOW_SIZE//2 or detected_fall_index > len(data) - WINDOW_SIZE//2:
              detected_falls.remove(detected_fall_index)
          
      data[:,0] = deltaTimes

      total_falls += len(actual_fall_times)
      true_positives = []
      false_positives = []
      # if a detected fall is within +-1500 ms of a true fall, count as true positive and remove the actual fall from the list to prevent double counting
      # otherwise count as false positive
      for detected_index in detected_falls:
          match_found = False
          window_data = data[max(0, detected_index - WINDOW_SIZE//2): min(len(data), detected_index + WINDOW_SIZE//2), :]
          for actual in actual_fall_times:
              if abs(times[detected_index] - actual) <= 1500:
                  true_positives.append(window_data)
                  actual_fall_times.remove(actual)
                  match_found = True
                  break
          if not match_found:
              false_positives.append(window_data)
              print(f"False positive detected at time {times[detected_index]}ms, in dataset {data_file}")
      all_true_positives.extend(true_positives)
      all_false_positives.extend(false_positives)

  print("True Positives:", len(all_true_positives))
  print("False Positives:", len(all_false_positives))
  print("Total Falls:", total_falls, "(Removed", total_real_falls_removed, "due to large time gaps)")

  return all_true_positives, all_false_positives, total_falls


def load_data():
  optimal_params = (6.6, 40, 16, 1.0, 0.07)
  true_positives, false_positives, total_falls = get_naive_classified_data(optimal_params)

  X = np.array(true_positives + false_positives)
  Y = np.array([1]*len(true_positives) + [0]*len(false_positives))


  X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
  X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42, stratify=Y_trainval)

  datasets = [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]

  for i in range(3):
    # Data augmentation: for each positive sample, create a mirrored version by negating acc_x and gyro_x
    # if (i == 0 or i == 1): # only for training set
    if (i == 0 or i == 1): # only for training set
      augmented_windows = list(datasets[i][0])
      augmented_labels = list(datasets[i][1])
      # for axis in range(3):
      #   for j in range(len(datasets[i][0])):
      #     window = datasets[i][0][j]
      #     label = datasets[i][1][j]
      #     mirrored_window = np.copy(window)
      #     mirrored_window[:, 1+axis] = -mirrored_window[:, 1+axis]  # Negate acc
      #     mirrored_window[:, 4+axis] = -mirrored_window[:, 4+axis]  # Negate gyro
      #     augmented_windows.append(mirrored_window)
      #     augmented_labels.append(label)
      for axis_slide in range(2):
        axis1 = axis_slide
        axis2 = (axis_slide + 1) % 3
        axis3 = (axis_slide + 2) % 3
        for j in range(len(datasets[i][0])):
          window = augmented_windows[j].copy()
          label = augmented_labels[j]
          window[:, 1+axis1], window[:, 1+axis2], window[:, 1+axis3] = window[:, 1+axis2], window[:, 1+axis3], window[:, 1+axis1]
          window[:, 4+axis1], window[:, 4+axis2], window[:, 4+axis3] = window[:, 4+axis2], window[:, 4+axis3], window[:, 4+axis1]
          augmented_windows.append(window)
          augmented_labels.append(label)

      datasets[i] = (np.array(augmented_windows), np.array(augmented_labels))

  X_train, Y_train = datasets[0]
  X_val, Y_val = datasets[1]
  X_test, Y_test = datasets[2]

  print(f"Train set: {sum(Y_train)} positive samples, {len(Y_train)-sum(Y_train)} negative samples")
  print(f"Val set: {sum(Y_val)} positive samples, {len(Y_val)-sum(Y_val)} negative samples")
  print(f"Test set: {sum(Y_test)} positive samples, {len(Y_test)-sum(Y_test)} negative samples")

  # Reshape labels to (N, 1) to match model output shape
  Y_train = Y_train.reshape(-1, 1)
  Y_val = Y_val.reshape(-1, 1)
  Y_test = Y_test.reshape(-1, 1)

  train_dataset = make_dataset(X_train, Y_train)
  val_dataset = make_dataset(X_val, Y_val)
  test_dataset = make_dataset(X_test, Y_test)
  # Get shape of one batch
  for x_batch, y_batch in train_dataset.take(1):
    print(f"One batch x shape: {x_batch.shape}, y shape: {y_batch.shape}")
    # Visualise window 1
    plt.figure(figsize=(12, 4))
    plt.plot(x_batch[0,:,1], label='acc_x')
    plt.plot(x_batch[0,:,2], label='acc_y')
    plt.plot(x_batch[0,:,3], label='acc_z')
    plt.plot(x_batch[0,:,4], label='gyro_x')
    plt.plot(x_batch[0,:,5], label='gyro_y')
    plt.plot(x_batch[0,:,6], label='gyro_z')
    plt.title(f'Sample Window Data (First in Batch) - Label: {y_batch[0].numpy()[0]}')
    plt.xlabel('Time Steps')
    plt.ylabel('Sensor Values')
    plt.legend()
    plt.show()

  for x_batch, y_batch in val_dataset.take(1):
    print(f"One batch x shape: {x_batch.shape}, y shape: {y_batch.shape}")

  return train_dataset, val_dataset, test_dataset

def make_dataset(x,y):
  def augment(window, label):
    # Add some random noise
    noise_factor = 0.08
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=window.shape).astype(np.float32)
    noisy_window = window + noise
    # Random scaling
    scale_factor = np.random.normal(loc=1.0, scale=0.1)
    noisy_window = noisy_window * scale_factor

    return noisy_window, label
  return tf.data.Dataset.from_tensor_slices((x,y)).map(augment).shuffle(len(x)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def get_index_after_time(data, time_ms, start_index):
  i = start_index
  while data[i][0] <= time_ms:
    i = i + 1
  return i

if __name__ == "__main__":
  # sort_csv()
  load_data()
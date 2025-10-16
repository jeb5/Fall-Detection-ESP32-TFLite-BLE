import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
import math


DATA_CSV = "data_sorted.csv"
FALL_CSV = "fall_video3_marks.csv"
CACHE_PATH = ".data_cache"

DATA_FILES = [["data/session1_data_sorted.csv", "data/session1_marks.csv"],
							["data/session2_data_sorted.csv", "data/session2_marks.csv"],
							["data/session3_data_sorted.csv", "data/session3_marks.csv"],
							["data/session4_data_sorted.csv", "data/session4_marks.csv"],
							["data/session5_data_sorted.csv", "data/session5_marks.csv"],
							["data/session6_data_sorted.csv", "data/session6_marks.csv"],
							]

FALL_WINDOW_MS = 3000
N_SAMPLES_EACH_FALL_TRAIN = 4
WINDOW_SIZE = 96
BATCH_SIZE = 32

np.set_printoptions(precision=4, suppress=True)

def load_data():
	X, Y = [], []

	for i, (data_file, fall_file) in enumerate(DATA_FILES):
		print(f"Processing {data_file} and {fall_file}...")
		data = np.loadtxt(data_file, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4, 5, 6, 7))
		falls = np.loadtxt(fall_file, delimiter=",", skiprows=1, usecols=(0))
		falls *= 1000  # convert to ms
		windows, labels = get_windows_and_labels(data, falls)
		X.extend(windows)
		Y.extend(labels)
	
	X_indices = np.arange(len(X))
	Y = np.array(Y)

	X_trainval, X_test, Y_trainval, Y_test = train_test_split(X_indices, Y, test_size=0.2, random_state=42, stratify=Y)
	X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42, stratify=Y_trainval)

	datasets = [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]
	# Print dataset shapes
	for i, (X_set, Y_set) in enumerate(datasets):
		print(f"Dataset {i} shapes: {X_set.shape}, {Y_set.shape}")
		print(f"Dataset {i} samples: {sum(Y_set)} positive, {len(Y_set)-sum(Y_set)} negative")

	for i, (X_set, Y_set) in enumerate(datasets):
		windows = []
		labels = []
		for j in range(len(X_set)):
			label, data, start_idx, end_idx = X[X_set[j]]
			if label == 0: # negative sample
				window = np.copy(data[start_idx:start_idx+WINDOW_SIZE])
				windows.append(window)
				labels.append([0])
			else: #positive sample
				# n = N_SAMPLES_EACH_FALL_TRAIN if (i == 0 or i == 1) else 1 # more samples for training set
				n = N_SAMPLES_EACH_FALL_TRAIN if (i == 0) else 1 # more samples for training set
				for _ in range(n):
					window_start = random.randrange(end_idx - WINDOW_SIZE, start_idx)
					window = np.copy(data[window_start:window_start + WINDOW_SIZE])
					windows.append(window)
					labels.append([1])

		# Loop through windows changing time to delta time, and removing windows with delta time > 100ms
		cleaned_windows = []
		cleaned_labels = []
		removed_count = 0
		for window, label in zip(windows, labels):
			delta_times = [window[i][0] - window[i-1][0] for i in range(1, len(window))]
			average_delta_time = math.floor(sum(delta_times) / len(delta_times))
			delta_times = [average_delta_time] + delta_times  # First delta time is average
			if any(dt < 10 for dt in delta_times):
				raise ValueError("Delta time less than 10ms found, data may be corrupted")
			if any(dt > 100 for dt in delta_times):
				removed_count += 1
				continue
			# create new window, with first column as delta times
			window[:, 0] = np.array(delta_times, dtype=window.dtype)

			cleaned_windows.append(window)
			cleaned_labels.append(label)
		print(f"Dataset {i}: Removed {removed_count} windows due to large/small time gaps")

		# for w in range(len(cleaned_windows)):
		# 	window = cleaned_windows[w]
		# 	quaternions= np.tile([1., 0., 0., 0.], (len(window), 1)) # Allocate for quaternions
		# 	madgwick = Madgwick()
		# 	for t in range(1, len(window)):
		# 		dt = window[t][0] / 1000.0  # convert ms to s
		# 		gyr = window[t][5:8]  # gyro x,y,z
		# 		acc = window[t][1:4]  # acc x,y,z
		# 		madgwick.Dt = dt
		# 		quat = madgwick.updateIMU(quaternions[t-1], gyr=gyr, acc=acc)
		# 		quaternions[t] = quat
		# 	# replace gyro data with quaternions
		# 	window = np.delete(window, [5,6,7], axis=1)
		# 	window = np.hstack((window, quaternions))
		# 	cleaned_windows[w] = window


		# Data augmentation: for each positive sample, create a mirrored version by negating acc_x and gyro_x
		# if (i == 0 or i == 1): # only for training set
		if (i == 0): # only for training set
			augmented_windows = []
			augmented_labels = []
			for axis in range(3):
				for window, label in zip(cleaned_windows, cleaned_labels):
					mirrored_window = np.copy(window)
					mirrored_window[:, 1+axis] = -mirrored_window[:, 1+axis]  # Negate acc
					mirrored_window[:, 4+axis] = -mirrored_window[:, 4+axis]  # Negate gyro
					augmented_windows.append(mirrored_window)
					augmented_labels.append(label)
			cleaned_windows.extend(augmented_windows)
			cleaned_labels.extend(augmented_labels)
			augmented_windows = []
			augmented_labels = []
			for axis_slide in range(2):
				axis1 = axis_slide
				axis2 = (axis_slide + 1) % 3
				axis3 = (axis_slide + 2) % 3
				for window, label in zip(cleaned_windows, cleaned_labels):
					slid_window = np.copy(window)
					slid_window[:, 1+axis1], slid_window[:, 1+axis2], slid_window[:, 1+axis3] = slid_window[:, 1+axis2], slid_window[:, 1+axis3], slid_window[:, 1+axis1]
					slid_window[:, 4+axis1], slid_window[:, 4+axis2], slid_window[:, 4+axis3] = slid_window[:, 4+axis2], slid_window[:, 4+axis3], slid_window[:, 4+axis1]
					augmented_windows.append(slid_window)
					augmented_labels.append(label)
			cleaned_windows.extend(augmented_windows)
			cleaned_labels.extend(augmented_labels)

		datasets[i] = (np.array(cleaned_windows), np.array(cleaned_labels))

	X_train, Y_train = datasets[0]
	X_val, Y_val = datasets[1]
	X_test, Y_test = datasets[2]

	print(f"Train set: {sum(Y_train)} positive samples, {len(Y_train)-sum(Y_train)} negative samples")
	print(f"Val set: {sum(Y_val)} positive samples, {len(Y_val)-sum(Y_val)} negative samples")
	print(f"Test set: {sum(Y_test)} positive samples, {len(Y_test)-sum(Y_test)} negative samples")

	train_dataset = make_dataset(X_train, Y_train)
	val_dataset = make_dataset(X_val, Y_val)
	test_dataset = make_dataset(X_test, Y_test)
	# Get shape of one batch
	for x_batch, y_batch in train_dataset.take(1):
		print(f"One batch x shape: {x_batch.shape}, y shape: {y_batch.shape}")
	for x_batch, y_batch in val_dataset.take(1):
		print(f"One batch x shape: {x_batch.shape}, y shape: {y_batch.shape}")

	return train_dataset, val_dataset, test_dataset


# def sort_csv():
# 	# Load data.csv: time,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
# 	data = np.loadtxt("data/session4_data.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

# 	# sort by time
# 	data = data[np.argsort(data[:, 0])]
# 	# save sorted data to data_sorted.csv, rounding floats to 6 decimal places
# 	np.savetxt("data/session4_data_sorted.csv", data, delimiter=",", fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"], header="time_ms,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z", comments='')


def get_windows_and_labels(data, falls):
	windows = []
	labels = []

	HFW = FALL_WINDOW_MS / 2  # Half fall window

	# Negative examples
	fall_index = 0
	next_fall_start = 0
	next_fall_end = 0
	i = 0
	while i < len(data):
		if next_fall_start - i > WINDOW_SIZE:
			windows.append((0,data,i,0))
			labels.append([0])
			i += WINDOW_SIZE
		else:
			i = next_fall_end
			fall_index += 1
			if fall_index < len(falls):
				next_fall_start = get_index_after_time(data, falls[fall_index] - HFW, next_fall_end)
				next_fall_end = get_index_after_time(data, falls[fall_index] + HFW, next_fall_start)
			else:
				next_fall_start = len(data) + 1  # No more falls
				next_fall_end = len(data) + 1
	# Positive examples
	for fall_time in falls:
		fall_index = get_index_after_time(data, fall_time, 0)
		start_index = get_index_after_time(data, fall_time - HFW, fall_index)
		end_index = get_index_after_time(data, fall_time + HFW, fall_index)
		if end_index - start_index >= WINDOW_SIZE:
			raise ValueError("Fall window is larger than sample window size")
		windows.append((1,data,start_index,end_index))
		labels.append([1])

	sum_labels = sum(label[0] for label in labels)

	print(f"Generated {len(windows)} windows, {sum_labels} positive samples, {len(labels)-sum_labels} negative samples")
	return windows, labels
	
def make_dataset(x,y):
	return tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(x)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def get_index_after_time(data, time_ms, start_index):
	i = start_index
	while data[i][0] <= time_ms:
		i = i + 1
	return i

if __name__ == "__main__":
	# sort_csv()
	load_data()
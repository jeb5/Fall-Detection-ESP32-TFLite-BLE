import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
import math
import os

DATA_CSV = "data_sorted.csv"
FALL_CSV = "fall_video3_marks.csv"
CACHE_PATH = ".data_cache"

DATA_FILES = [["data/session1_data_sorted.csv", "data/session1_marks.csv"],
							["data/session2_data_sorted.csv", "data/session2_marks.csv"],
							["data/session3_data_sorted.csv", "data/session3_marks.csv"],
							["data/session4_data_sorted.csv", "data/session4_marks.csv"]]

# TODO: Prevent the same fall from being in both train and test sets, while keeping the N_SAMPLES_EACH_FALL data augmentation.

FALL_WINDOW_MS = 3000
N_SAMPLES_EACH_FALL = 1
WINDOW_SIZE = 64
BATCH_SIZE = 32

np.set_printoptions(precision=4, suppress=True)

def load_data():
	X, Y = [], []

	os.makedirs(CACHE_PATH, exist_ok=True)
	cache_mtime = os.path.getmtime(os.path.join(CACHE_PATH, "data.npz")) if os.path.exists(os.path.join(CACHE_PATH, "data.npz")) else 0
	data_mtimes = [os.path.getmtime(f[0]) for f in DATA_FILES] + [os.path.getmtime(f[1]) for f in DATA_FILES]
	if cache_mtime > max(data_mtimes):
		print("Loading datasets from cache...")
		cache = np.load(os.path.join(CACHE_PATH, "data.npz"))
		X = cache["X"]
		Y = cache["Y"]
	else:
		print("Preparing datasets from raw data...")

		for i, (data_file, fall_file) in enumerate(DATA_FILES):
			print(f"Processing {data_file} and {fall_file}...")
			data = np.loadtxt(data_file, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4, 5, 6, 7))
			falls = np.loadtxt(fall_file, delimiter=",", skiprows=1, usecols=(0))
			falls *= 1000  # convert to ms
			windows, labels = get_windows_and_labels(data, falls)
			X.extend(windows)
			Y.extend(labels)
		
		X = np.array(X)
		print(f"X shape: {X.shape}")
		Y = np.array(Y)
		print(f"Y shape: {Y.shape}")

		# Save X and Y to cache
		np.savez_compressed(os.path.join(CACHE_PATH, "data.npz"), X=X, Y=Y)

	X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
	X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42, stratify=Y_trainval)

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


def sort_csv():
	# Load data.csv: time,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
	data = np.loadtxt("data/session4_data.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

	# sort by time
	data = data[np.argsort(data[:, 0])]
	# save sorted data to data_sorted.csv, rounding floats to 6 decimal places
	np.savetxt("data/session4_data_sorted.csv", data, delimiter=",", fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"], header="time_ms,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z", comments='')


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
			window = np.copy(data[i:i+WINDOW_SIZE])
			windows.append(window)
			labels.append([0])
			i += WINDOW_SIZE
			# print (f"{i} -> {i + WINDOW_SIZE} : 0")
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
		for _ in range(N_SAMPLES_EACH_FALL):
			window_start = random.randrange(end_index - WINDOW_SIZE, start_index)
			window = np.copy(data[window_start:window_start + WINDOW_SIZE])
			windows.append(window)
			labels.append([1])
			# print (f"{window_start} -> {window_start + WINDOW_SIZE} : 1")
	

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
	print(f"Removed {removed_count} windows due to large/small time gaps")


	return cleaned_windows, cleaned_labels

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
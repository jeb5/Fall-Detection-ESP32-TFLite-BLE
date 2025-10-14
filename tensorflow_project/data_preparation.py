import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
import os

DATA_CSV = "data_sorted.csv"
FALL_CSV = "fall_video3_marks.csv"
DATASET_PATH = ".dataset_cache/"

FALL_WINDOW_MS = 3000
N_SAMPLES_EACH_FALL = 5
WINDOW_SIZE = 64
BATCH_SIZE = 32

def load_data():

	os.makedirs(DATASET_PATH, exist_ok=True)
	# Check for saved datasets, and if they were last modified after data.csv or fall_video3_marks.csv
	if (os.path.exists(os.path.join(DATASET_PATH, "train")) and os.path.exists(os.path.join(DATASET_PATH, "val")) and os.path.exists(os.path.join(DATASET_PATH, "test"))):
		dataset_mtime = min(os.path.getmtime(os.path.join(DATASET_PATH, "train")), os.path.getmtime(os.path.join(DATASET_PATH, "val")), os.path.getmtime(os.path.join(DATASET_PATH, "test")))
		data_mtime = os.path.getmtime(DATA_CSV)
		fall_mtime = os.path.getmtime(FALL_CSV)
		if dataset_mtime > data_mtime and dataset_mtime > fall_mtime:
			print("Loading datasets from disk...")
			train_dataset = tf.data.Dataset.load(os.path.join(DATASET_PATH, "train"))
			val_dataset = tf.data.Dataset.load(os.path.join(DATASET_PATH, "val"))
			test_dataset = tf.data.Dataset.load(os.path.join(DATASET_PATH, "test"))
			return train_dataset, val_dataset, test_dataset

	print("Preparing datasets from raw data...")

	data = np.loadtxt(DATA_CSV, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4, 5, 6, 7))
	falls = np.loadtxt(FALL_CSV, delimiter=",", skiprows=1, usecols=(0))
	falls *= 1000  # convert to ms
	X, Y = get_windows_and_labels(data, falls)
	X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
	X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42, stratify=Y_trainval)

	print(f"Train set: {sum(Y_train)} positive samples, {len(Y_train)-sum(Y_train)} negative samples")
	print(f"Val set: {sum(Y_val)} positive samples, {len(Y_val)-sum(Y_val)} negative samples")
	print(f"Test set: {sum(Y_test)} positive samples, {len(Y_test)-sum(Y_test)} negative samples")

	train_dataset = make_dataset(X_train, Y_train)
	val_dataset = make_dataset(X_val, Y_val)
	test_dataset = make_dataset(X_test, Y_test)

	# Save datasets
	tf.data.Dataset.save(train_dataset, os.path.join(DATASET_PATH, "train"))
	tf.data.Dataset.save(val_dataset, os.path.join(DATASET_PATH, "val"))
	tf.data.Dataset.save(test_dataset, os.path.join(DATASET_PATH, "test"))
	return train_dataset, val_dataset, test_dataset


def sort_csv():
	# Load data.csv: time,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
	data = np.loadtxt("data.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

	# sort by time
	data = data[np.argsort(data[:, 0])]
	# save sorted data to data_sorted.csv, rounding floats to 6 decimal places
	np.savetxt("data_sorted.csv", data, delimiter=",", fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"], header="time_ms,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z", comments='')


def get_windows_and_labels(data, falls):
	windows = np.empty((0, WINDOW_SIZE, 6))

	labels = np.empty((0, 1))
	HFW = FALL_WINDOW_MS / 2  # Half fall window

	# Negative examples
	fall_index = 0
	next_fall_start = 0
	next_fall_end = 0
	i = 0
	while i < len(data):
		if next_fall_start - i > WINDOW_SIZE:
			window = data[i:i+WINDOW_SIZE]
			windows = np.append(windows, [window[:, 1:]], axis=0)
			labels = np.append(labels, [[0]], axis=0)
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
			window = data[window_start:window_start + WINDOW_SIZE]
			windows = np.append(windows, [window[:, 1:]], axis=0)
			labels = np.append(labels, [[1]], axis=0)
			# print (f"{window_start} -> {window_start + WINDOW_SIZE} : 1")
	
	return windows, labels

def make_dataset(x,y):
	return tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(x)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def get_index_after_time(data, time_ms, start_index):
	i = start_index
	while data[i][0] <= time_ms:
		i = i + 1
	return i

if __name__ == "__main__":
	load_data()
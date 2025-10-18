from skopt import gp_minimize, forest_minimize
from skopt.space import Integer, Real
from skopt.plots import plot_convergence, plot_objective
import numpy as np
import matplotlib.pyplot as plt
from naive import custom_feature

DATA_FILES = [
        ["data/session1_data_sorted.csv", "data/session1_marks.csv"],
        ["data/session2_data_sorted.csv", "data/session2_marks.csv"],
        ["data/session3_data_sorted.csv", "data/session3_marks.csv"],
        ["data/session4_data_sorted.csv", "data/session4_marks.csv"],
        ["data/session5_data_sorted.csv", "data/session5_marks.csv"],
        ["data/session6_data_sorted.csv", "data/session6_marks.csv"],
        ["data/session8_data_sorted.csv", "data/session8_marks.csv"],
        ]
experiment_number = 0
experiment_results = []
def f(params):
    global experiment_number
    global experiment_results
    experiment_number += 1
    print(f"Experiment {experiment_number} with params: {params}")
    dog_sigma, acc_lp_window, angle_lp_window, angle_exp, threshold = params
    acc_lp_window *= 2  # make even
    angle_lp_window *= 2  # make even
    total_true_positives = 0
    total_false_positives = 0
    total_falls = 0
    total_real_falls_removed = 0
    for i in range(len(DATA_FILES)):
        data_file, fall_file = DATA_FILES[i]
        # Read fall file. 1 column, skip header
        falls_ms = np.loadtxt(fall_file, delimiter=",", skiprows=1, usecols=(0))  * 1000  # convert to ms
        # Read data file. 7 columns, skip header (time,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
        # Load only time, acc_x, acc_y, acc_z (cols 2,3,4)
        data = np.loadtxt(data_file, delimiter=",", skiprows=1, usecols=(0,2,3,4))
        times = data[:,0]  # shape (N,)
        deltaTimes = np.diff(times, prepend=0)  # shape (N,)
        acc_data = data[:,1:4].T  # shape (3, N)
        custom_feature_data = custom_feature(acc_data, dog_sigma, acc_lp_window, angle_lp_window, angle_exp)
        # Find the middle of each block, where custom_feature_data > threshold
        blocks = np.where(np.diff(np.concatenate(([0], custom_feature_data > threshold, [0]))))[0].reshape(-1, 2)
        # Get time of maximum in each block
        detected_falls = []
        for start, end in blocks:
            if end > start:
                max_index = np.argmax(custom_feature_data[start:end]) + start
                detected_falls.append(times[max_index])
        
        actual_falls = falls_ms.tolist()

        # For each actual fall, if there is a deltaTime >200ms within +-96 indicies of it, remove it
        # For each detected fall, if there is a deltaTime >200ms within +-96 indicies of it, remove it
        suspicious_window = 60
        suspicious_delta_time_threshold = 300  # ms
        actual_falls_original = actual_falls.copy()
        for fall_time in actual_falls_original:
            fall_index = np.searchsorted(times, fall_time)
            start_index = max(0, fall_index - suspicious_window)
            end_index = min(len(deltaTimes), fall_index + suspicious_window)
            if np.any(deltaTimes[start_index:end_index] > suspicious_delta_time_threshold):
                actual_falls.remove(fall_time)
        total_real_falls_removed += len(actual_falls_original) - len(actual_falls)
        detected_falls_original = detected_falls.copy()
        for detected_time in detected_falls_original:
            detected_index = np.searchsorted(times, detected_time)
            start_index = max(0, detected_index - suspicious_window)
            end_index = min(len(deltaTimes), detected_index + suspicious_window)
            if np.any(deltaTimes[start_index:end_index] > suspicious_delta_time_threshold):
                detected_falls.remove(detected_time)
            
        total_falls += len(actual_falls)
        true_positives = 0
        false_positives = 0
        # if a detected fall is within +-1500 ms of a true fall, count as true positive and remove the actual fall from the list to prevent double counting
        # otherwise count as false positive
        for detected in detected_falls:
            match_found = False
            for actual in actual_falls:
                if abs(detected - actual) <= 1500:
                    true_positives += 1
                    actual_falls.remove(actual)
                    match_found = True
                    break
            if not match_found:
                false_positives += 1
        total_true_positives += true_positives
        total_false_positives += false_positives

    print("True Positives:", total_true_positives)
    print("False Positives:", total_false_positives)
    print("Total Falls:", total_falls, "(Removed", total_real_falls_removed, "due to large time gaps)")
    experiment_results.append((params, total_true_positives, total_false_positives))

    score = (total_falls - total_true_positives + 5) * (150 + total_false_positives)
    return score



# --- Define parameter space ---
space = [
    Real(5.0, 15.0, name='dog_sigma'),
    Integer(3, 30, name='angle_lp_window'),
    Integer(3, 30, name='acc_lp_window'),
    Real(0.3, 1.5, name='angle_exp'),
    Real(0.01, 0.2, name='threshold')
]


# # --- Run Bayesian optimization ---
result = gp_minimize(
    func=f,
    dimensions=space,
    n_calls=100,           # number of function evaluations
    n_initial_points=10,  # initial random points before modeling
    random_state=42
)
# --- Run Random Forest based optimization ---
# result = forest_minimize(
#     func=f,
#     dimensions=space,
#     n_calls=200,           # number of function evaluations
#     n_initial_points=10,  # initial random points before modeling
#     random_state=42
# )

# --- Show best parameters ---
best_params = result.x
best_value = result.fun
best_index = result.func_vals.argmin()
print(f"Best experiment number: {best_index + 1}")
print("Best parameters:", best_params)
print("Minimised function value:", best_value)

print(f"Best experiment results: {experiment_results[best_index]}")

# --- Visualize optimization progress ---
plot_convergence(result)
plt.show()

# --- Visualize parameter importance and pairwise effects ---
_ = plot_objective(result)
plt.show()

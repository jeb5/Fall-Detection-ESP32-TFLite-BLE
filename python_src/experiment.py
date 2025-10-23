from skopt import gp_minimize, forest_minimize
from skopt.space import Integer, Real
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt
from data_preparation import get_naive_classified_data

experiment_number = 0
experiment_results = []
def f(params):
    global experiment_number
    global experiment_results
    experiment_number += 1

    dog_sigma, acc_lp_window, angle_lp_window, threshold = params
    acc_lp_window *= 2  # make even
    angle_lp_window *= 2  # make even
    angle_exp = 1.0  # fixed value
    new_params = (dog_sigma, acc_lp_window, angle_lp_window, angle_exp, threshold)

    print(f"Experiment {experiment_number} with params: {new_params}")
    all_true_positives, all_false_positives, total_real_falls = get_naive_classified_data(new_params)

    experiment_results.append((new_params, len(all_true_positives), len(all_false_positives)))

    score = (total_real_falls - len(all_true_positives) + 10) * (150 + len(all_false_positives))
    return score




# --- Define parameter space ---
space = [
    Real(5.0, 15.0, name='dog_sigma'),
    Integer(1, 40, name='angle_lp_window'),
    Integer(1, 30, name='acc_lp_window'),
    # Real(0.3, 1.5, name='angle_exp'),
    Real(0.01, 0.2, name='threshold')
]


# # --- Run Bayesian optimization ---
result = gp_minimize(
    func=f,
    dimensions=space,
    n_calls=200,           # number of function evaluations
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

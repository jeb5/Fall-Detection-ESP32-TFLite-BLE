import math
import numpy as np
def detect_fall(window):
    pass

def compute_dog_kernel(sigma_small, sigma_large):
        radius = int(np.ceil(3 * sigma_large))
        x = np.arange(-radius, radius + 1)

        gauss_small = np.exp(-0.5 * (x / sigma_small) ** 2)
        gauss_small /= gauss_small.sum()

        gauss_large = np.exp(-0.5 * (x / sigma_large) ** 2)
        gauss_large /= gauss_large.sum()

        dog_kernel = gauss_small - gauss_large
        return dog_kernel
dog_kernel = compute_dog_kernel(sigma_small=9, sigma_large=27)
    

# Takes a window of size 192?
def naive(window):
    global dog_kernel
    # - acc_mag
    # - acc_mag dog (sigma 9 and 27, radius 163)
    # - angle_from_lp (30 window LP on each channel)
    # - angle_from_lp lp (30 window)
    # combined feature (*)
    # computing custom feature at -DoG_size//2
    fi = - len(dog_kernel) // 2 # =
    dog_value = 0.0
    for i, kernel_value in enumerate(dog_kernel):
        acc_mag = math.sqrt(window["acc_x"][-i]**2 + window["acc_y"][-i]**2 + window["acc_z"][-i]**2)
        dog_value += acc_mag * kernel_value
    
    ax_sum = 0.0
    ay_sum = 0.0
    az_sum = 0.0
    for i in range(30):
        ax_sum += window["acc_x"][fi + i - 30]
        ay_sum += window["acc_y"][fi + i - 30]
        az_sum += window["acc_z"][fi + i - 30]
    angle_sum = 0.0
    for i in range(30):
        ax_lp = ax_sum / 30.0
        ay_lp = ay_sum / 30.0
        az_lp = az_sum / 30.0
        ax_sum += window["acc_x"][fi + i]
        ay_sum += window["acc_y"][fi + i]
        az_sum += window["acc_z"][fi + i]
        ax_sum -= window["acc_x"][fi + i - 30]
        ay_sum -= window["acc_y"][fi + i - 30]
        az_sum -= window["acc_z"][fi + i - 30]
        ax = window["acc_x"][fi + i - 15]
        ay = window["acc_y"][fi + i - 15]
        az = window["acc_z"][fi + i - 15]
        dot_product = ax * ax_lp + ay * ay_lp + az * az_lp
        mag_orig = math.sqrt(ax**2 + ay**2 + az**2)
        mag_lp = math.sqrt(ax_lp**2 + ay_lp**2 + az_lp**2)
        cos_angle = dot_product / (mag_orig * mag_lp + 1e-6)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        angle_sum += angle

    average_angle = angle_sum / 30.0
    feature = dog_value * average_angle
    # feature = average_angle
    return feature

        # if all(k in self.data_v_dict for k in ["acc_x", "acc_y", "acc_z"]):
        #     acc_x = self.data_v_dict["acc_x"]
        #     acc_y = self.data_v_dict["acc_y"]
        #     acc_z = self.data_v_dict["acc_z"]
        #     acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        #     self.custom_data_dict["acc_magnitude"] = acc_mag
            
        #     # Create plot item for acceleration magnitude
        #     pen = pg.mkPen(color="g", width=1)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Acc Magnitude")
        #     self.custom_plot_items.append(item)
        
        # # Acceleration magnitude low-pass filtered
        # if "acc_magnitude" in self.custom_data_dict:
        #     acc_mag = self.custom_data_dict["acc_magnitude"]

        #     window_size = 96  # samples
        #     # Simple averaging filter
        #     kernel = np.ones(window_size) / window_size
        #     acc_mag_lp = np.convolve(acc_mag, kernel, mode="same")

        #     self.custom_data_dict["acc_magnitude_lp"] = acc_mag_lp
            
        #     # Create plot item for low-pass filtered acceleration magnitude
        #     pen = pg.mkPen(color="r", width=1)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Acc Magnitude LPF")
        #     self.custom_plot_items.append(item)

        # # Difference of Gaussians on acceleration magnitude
        # if "acc_magnitude" in self.custom_data_dict:
        #     acc_mag = self.custom_data_dict["acc_magnitude"]

        #     sigma_small = 9
        #     sigma_large = sigma_small * 3
        #     radius = int(np.ceil(3 * sigma_large))
        #     x = np.arange(-radius, radius + 1)

        #     gauss_small = np.exp(-0.5 * (x / sigma_small) ** 2)
        #     gauss_small /= gauss_small.sum()

        #     gauss_large = np.exp(-0.5 * (x / sigma_large) ** 2)
        #     gauss_large /= gauss_large.sum()

        #     dog_kernel = gauss_small - gauss_large
        #     acc_mag_dog = np.convolve(acc_mag, dog_kernel, mode="same")
        #     self.custom_data_dict["acc_magnitude_dog"] = acc_mag_dog
            
        #     # Create plot item for low-pass filtered acceleration magnitude
        #     pen = pg.mkPen(color="b", width=2)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Acc Magnitude DoG")
        #     self.custom_plot_items.append(item)
        # # Angle away from low pass acceleration
        # if "acc_magnitude_lp" in self.custom_data_dict:
        #     # Get low-pass acc_x, acc_y, acc_z
        #     acc_x = self.data_v_dict["acc_x"]
        #     acc_y = self.data_v_dict["acc_y"]
        #     acc_z = self.data_v_dict["acc_z"]
        #     # window_size = 96  # samples
        #     window_size = 30  # samples
        #     kernel = np.ones(window_size) / window_size
        #     acc_x_lp = np.convolve(acc_x, kernel, mode="same")
        #     acc_y_lp = np.convolve(acc_y, kernel, mode="same")
        #     acc_z_lp = np.convolve(acc_z, kernel, mode="same")
        #     # Get angle between (acc_x, acc_y, acc_z) and (acc_x_lp, acc_y_lp, acc_z_lp)
        #     dot_product = (acc_x * acc_x_lp + acc_y * acc_y_lp + acc_z * acc_z_lp)
        #     mag_orig = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        #     mag_lp = np.sqrt(acc_x_lp**2 + acc_y_lp**2 + acc_z_lp**2)
        #     cos_angle = dot_product / (mag_orig * mag_lp + 1e-6)
        #     cos_angle = np.clip(cos_angle, -1.0, 1.0)
        #     angles = np.arccos(cos_angle)  # in radians
        #     self.custom_data_dict["angle_from_lp"] = angles
        #     # Create plot item for angle from low-pass acceleration
        #     pen = pg.mkPen(color="c", width=0.5)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Angle from LP Acc")
        #     self.custom_plot_items.append(item)

        # # Low-pass angle change
        # if "angle_from_lp" in self.custom_data_dict:
        #     angles = self.custom_data_dict["angle_from_lp"]
        #     window_size = 30  # samples
        #     kernel = np.ones(window_size) / window_size
        #     angle_diff_lp = np.convolve(angles, kernel, mode="same")
        #     self.custom_data_dict["angle_change_lp"] = np.abs(angle_diff_lp)
            
        #     # Create plot item for low-pass angle change
        #     pen = pg.mkPen(color="y", width=1)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Angle Change LPF")
        #     self.custom_plot_items.append(item)
        
        # # low pass angle change multiplied by acc dog
        # if "angle_change_lp" in self.custom_data_dict and "acc_magnitude_dog" in self.custom_data_dict:
        #     angle_change_lp = self.custom_data_dict["angle_change_lp"]
        #     acc_mag_dog = self.custom_data_dict["acc_magnitude_dog"]
        #     combined_feature = angle_change_lp * acc_mag_dog
        #     self.custom_data_dict["combined_feature"] = combined_feature
            
        #     # Create plot item for combined feature
        #     pen = pg.mkPen(color="w", width=2)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Combined Feature")
        #     self.custom_plot_items.append(item)

        # naive_values = [0] * (193 - 81)
        # for i in range(len(self.data_t)-193):
        #     input_to_naive = {
        #         "acc_x": self.data_v_dict["acc_x"][i:i+193],
        #         "acc_y": self.data_v_dict["acc_y"][i:i+193],
        #         "acc_z": self.data_v_dict["acc_z"][i:i+193],
        #     }
        #     possible_feature = naive(input_to_naive)
        #     naive_values.append(possible_feature)
        # naive_values += [0] * (len(self.data_t) - len(naive_values)) 
        # self.custom_data_dict["naive_feature"] = np.array(naive_values)
        # # Create plot item for naive feature
        # pen = pg.mkPen(color="m", width=2)
        # item = self.plot_custom.plot([], [], pen=pen, name="Naive Feature")
        # self.custom_plot_items.append(item)

        # if "combined_feature" in self.custom_data_dict:
        #     fall = self.custom_data_dict["combined_feature"] > 0.07
        #     self.custom_data_dict["fall_events"] = fall.astype(float)
        #     # Create plot item for fall events
        #     pen = pg.mkPen(color="r", width=5)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Fall Events")
        #     self.custom_plot_items.append(item)
def custom_feature(data, dog_sigma, acc_lp_window, angle_lp_window, angle_exp):
    # Data is np array with columns: acc_x, acc_y, acc_z
    acc_mag = np.sqrt(data[0,:]**2 + data[1,:]**2 + data[2,:]**2)

    large_sigma = dog_sigma * 3
    radius = int(np.ceil(3 * large_sigma))
    x = np.arange(-radius, radius + 1)
    gauss_small = np.exp(-0.5 * (x / dog_sigma) ** 2)
    gauss_small /= gauss_small.sum()
    gauss_large = np.exp(-0.5 * (x / large_sigma) ** 2)
    gauss_large /= gauss_large.sum()
    dog_kernel = gauss_small - gauss_large
    acc_mag_dog = np.convolve(acc_mag, dog_kernel, mode="same")
    acc_x_lp = np.convolve(data[0,:], np.ones(acc_lp_window)/acc_lp_window, mode="same")
    acc_y_lp = np.convolve(data[1,:], np.ones(acc_lp_window)/acc_lp_window, mode="same")
    acc_z_lp = np.convolve(data[2,:], np.ones(acc_lp_window)/acc_lp_window, mode="same")
    dot_product = (data[0,:] * acc_x_lp + data[1,:] * acc_y_lp + data[2,:] * acc_z_lp)
    mag_orig = np.sqrt(data[0,:]**2 + data[1,:]**2 + data[2,:]**2)
    mag_lp = np.sqrt(acc_x_lp**2 + acc_y_lp**2 + acc_z_lp**2)
    cos_angle = dot_product / (mag_orig * mag_lp + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)  # in radians
    angles_lp = np.convolve(angles, np.ones(angle_lp_window)/angle_lp_window, mode="same")
    combined_feature = (angles_lp ** angle_exp) * acc_mag_dog
    return combined_feature

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt

def plot_data(x, y, xlabel, ylabel, title, legend=None):
    plt.figure()
    plt.plot(x, y, label=legend)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()
def wrap_to_pi(angle):
    angle = np.remainder(angle, 2 * np.pi)
    angle[angle > np.pi] -= 2 * np.pi
    return angle
def butter_filter(data, cutoff, fs, btype='low', order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff <= 0 or normal_cutoff >= 1:
        raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1")
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, data)

# File paths
circle_gps_path = '/home/sha/Docs/northeastern/RSN/LAB5/D3/data_going_in_circles_gps_data.csv'
circle_imu_path = '/home/sha/Docs/northeastern/RSN/LAB5/D3/data_going_in_circles_imu.csv'
drive_gps_path = '/home/sha/Docs/northeastern/RSN/LAB5/D3/data_driving_gps_data.csv'
drive_imu_path = '/home/sha/Docs/northeastern/RSN/LAB5/D3/data_driving_imu.csv'

# Load CSVs
circle_gps = pd.read_csv(circle_gps_path)
circle_imu = pd.read_csv(circle_imu_path)
drive_gps = pd.read_csv(drive_gps_path)
drive_imu = pd.read_csv(drive_imu_path)

# Sort and drop duplicates
circle_imu = circle_imu.sort_values('header.stamp.secs').drop_duplicates('header.stamp.secs')
drive_imu = drive_imu.sort_values('header.stamp.secs').drop_duplicates('header.stamp.secs')
circle_gps = circle_gps.sort_values('header.stamp.secs').drop_duplicates('header.stamp.secs')
drive_gps = drive_gps.sort_values('header.stamp.secs').drop_duplicates('header.stamp.secs')

# Extract relevant columns
circle_imu = circle_imu[['header.stamp.secs', 'imu.angular_velocity.z', 'imu.linear_acceleration.x','imu.linear_acceleration.y', 'mag_field.magnetic_field.x', 'mag_field.magnetic_field.y']]
drive_imu = drive_imu[['header.stamp.secs', 'imu.angular_velocity.z', 'imu.linear_acceleration.x','imu.linear_acceleration.y', 'mag_field.magnetic_field.x', 'mag_field.magnetic_field.y']]
circle_gps = circle_gps[['header.stamp.secs', 'utm_easting', 'utm_northing']]
drive_gps = drive_gps[['header.stamp.secs', 'utm_easting', 'utm_northing']]

# Magnetometer Calibration
Mx_raw = circle_imu['mag_field.magnetic_field.x'].values
My_raw = circle_imu['mag_field.magnetic_field.y'].values
Mx_mean, My_mean = Mx_raw.mean(), My_raw.mean()
Mx_corrected = Mx_raw - Mx_mean
My_corrected = My_raw - My_mean
scale_factor = (Mx_corrected.std() + My_corrected.std()) / 2
Mx_cal = Mx_corrected / Mx_corrected.std() * scale_factor
My_cal = My_corrected / My_corrected.std() * scale_factor
plt.figure()
plt.scatter(Mx_raw, My_raw, label='Raw Magnetometer Data', alpha=0.5)
plt.scatter(Mx_cal, My_cal, label='Calibrated Magnetometer Data', alpha=0.5)
plt.xlabel('Magnetic Field X')
plt.ylabel('Magnetic Field Y')
plt.legend()
plt.title('Magnetometer Data Before and After Calibration')
plt.grid(True)
plt.show()

# Magnetometer Yaw Estimation
# Ensure consistent lengths
min_length = min(len(circle_imu['header.stamp.secs']), len(Mx_raw))
time_circle = circle_imu['header.stamp.secs'].values[:min_length]
Mx_raw = Mx_raw[:min_length]
My_raw = My_raw[:min_length]
Mx_cal = Mx_cal[:min_length]
My_cal = My_cal[:min_length]
yaw_mag = np.arctan2(My_cal, Mx_cal)
plt.figure()
plt.plot(time_circle, np.arctan2(My_raw, Mx_raw), label='Raw Magnetometer Yaw')
plt.plot(time_circle, yaw_mag, label='Calibrated Magnetometer Yaw')
plt.xlabel('Time (s)')
plt.ylabel('Yaw (rad)')
plt.legend()
plt.title('Magnetometer Yaw Before and After Calibration')
plt.grid(True)
plt.show()

# Gyro Yaw Estimation
Gz_circle = circle_imu['imu.angular_velocity.z'].values[:min_length]
yaw_gyro = cumulative_trapezoid(Gz_circle, time_circle, initial=0.0)
plt.figure()
plt.plot(time_circle, yaw_gyro, label='Gyro Yaw Estimation')
plt.xlabel('Time (s)')
plt.ylabel('Yaw (rad)')
plt.title('Gyro Yaw Estimation Over Time')
plt.grid(True)
plt.legend()
plt.show()

# Complementary Filter
fs = 1 / np.mean(np.diff(time_circle))
yaw_mag_filtered = butter_filter(yaw_mag, cutoff=0.1, fs=fs, btype='low')
yaw_gyro_filtered = butter_filter(yaw_gyro, cutoff=0.1, fs=fs, btype='high')
yaw_combined = wrap_to_pi(yaw_mag_filtered + yaw_gyro_filtered)
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(time_circle, yaw_mag_filtered, label='Low-Pass Magnetometer Yaw')
plt.title('Low-Pass Filtered Magnetometer Yaw')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(time_circle, yaw_gyro_filtered, label='High-Pass Gyro Yaw')
plt.title('High-Pass Filtered Gyro Yaw')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(time_circle, yaw_combined, label='Complementary Filter Output')
plt.title('Complementary Filter Output')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(time_circle, yaw_mag_filtered + yaw_gyro_filtered, label='IMU Heading Estimate')
plt.title('IMU Heading Estimate')
plt.xlabel('Time (s)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Velocity Estimation
Ax_drive = drive_imu['imu.linear_acceleration.x'].values
Ax_bias = Ax_drive[:100].mean()  # Assume first 100 samples are stationary
Ax_corrected = Ax_drive - Ax_bias
# Clean up timestamps to ensure proper sampling frequency computation
time_drive = drive_imu['header.stamp.secs'].values
valid_diff = np.diff(time_drive) > 0
if not np.all(valid_diff):
    time_drive = np.unique(time_drive)  # Ensure uniqueness
fs = 1 / np.mean(np.diff(time_drive))  # Recompute sampling frequency after cleaning
fs = max(fs, 2)  # Ensure fs is high enough for meaningful filtering
Ax_filtered = butter_filter(Ax_corrected, cutoff=0.5, fs=fs, btype='low')
# Ensure consistent timestamp lengths for plotting
min_length_drive = min(len(time_drive), len(Ax_drive))
Ax_drive = Ax_drive[:min_length_drive]
Ax_corrected = Ax_corrected[:min_length_drive]
Ax_filtered = Ax_filtered[:min_length_drive]
velocity = cumulative_trapezoid(Ax_filtered, time_drive[:min_length_drive], initial=0.0)
plt.figure()
plt.plot(time_drive[:min_length_drive], Ax_drive, label='Raw Acceleration')
plt.plot(time_drive[:min_length_drive], Ax_corrected, label='Bias-Corrected Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Forward Acceleration Before and After Adjustment')
plt.grid(True)
plt.legend()
plt.show()
plt.figure()
plt.plot(time_drive[:len(velocity)], velocity, label='Forward Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Forward Velocity from Accelerometer')
plt.grid(True)
plt.legend()
plt.show()
# GPS Velocity Estimation
gps_velocity = np.gradient(drive_gps['utm_easting'].to_numpy(), drive_gps['header.stamp.secs'].to_numpy())
plt.figure()
plt.plot(drive_gps['header.stamp.secs'].to_numpy(), gps_velocity, label='GPS Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Forward Velocity from GPS')
plt.grid(True)
plt.legend()
plt.show()

# Trajectory Estimation
yaw_combined_interp = np.interp(drive_imu['header.stamp.secs'].to_numpy(), time_circle, yaw_combined)
v_e = velocity * np.cos(yaw_combined_interp)
v_n = velocity * np.sin(yaw_combined_interp)
position_e = cumulative_trapezoid(v_e, time_drive[:len(v_e)], initial=0.0)
position_n = cumulative_trapezoid(v_n, time_drive[:len(v_n)], initial=0.0)
# Align trajectory to GPS starting point
gps_start_e = drive_gps['utm_easting'].iloc[0]
gps_start_n = drive_gps['utm_northing'].iloc[0]
position_e += gps_start_e
position_n += gps_start_n
plt.figure()
plt.plot(drive_gps['utm_easting'].to_numpy(), drive_gps['utm_northing'].to_numpy(), label='GPS Trajectory', linestyle='--')
plt.plot(position_e, position_n, label='IMU Dead Reckoning Trajectory')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Trajectory Comparison')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
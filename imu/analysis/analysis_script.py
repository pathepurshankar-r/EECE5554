import pandas as pd
from bagpy import bagreader
import matplotlib.pyplot as plt
import numpy as np
import re
import allantools as at

bag = bagreader('/home/sha/Docs/northeastern/RSN/LAB4/Analysis/imu_stationary.bag')
imu_data = bag.message_by_topic(topic='/imu')
data = pd.read_csv(imu_data)
gyro_x = data['imu.angular_velocity.x']
gyro_y = data['imu.angular_velocity.y']
gyro_z = data['imu.angular_velocity.z']
accel_x = data['imu.linear_acceleration.x']
accel_y = data['imu.linear_acceleration.y']
accel_z = data['imu.linear_acceleration.z']
rotation_x = data['imu.orientation.x']
rotation_y = data['imu.orientation.y']
rotation_z = data['imu.orientation.z']

# 1. Plot rotational rate from the gyro in degrees/s on axes x, y, z (Fig 0)
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(gyro_x, marker='o', linestyle='-', label='Gyro X')
plt.xlabel('Sample')
plt.ylabel('Degrees/s')
plt.title('Rotational Rate - Gyro X')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(gyro_y, marker='x', linestyle='-', label='Gyro Y', color='orange')
plt.xlabel('Sample')
plt.ylabel('Degrees/s')
plt.title('Rotational Rate - Gyro Y')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(gyro_z, marker='s', linestyle='-', label='Gyro Z', color='green')
plt.xlabel('Sample')
plt.ylabel('Degrees/s')
plt.title('Rotational Rate - Gyro Z')
plt.legend()

plt.tight_layout()
plt.show()

# 2. Plot acceleration from the accelerometer in m/s² on axes x, y, z (Fig 1)
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(accel_x, marker='o', linestyle='-', label='Accel X')
plt.xlabel('Sample')
plt.ylabel('m/s²')
plt.title('Acceleration - Accelerometer X')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(accel_y, marker='x', linestyle='-', label='Accel Y', color='orange')
plt.xlabel('Sample')
plt.ylabel('m/s²')
plt.title('Acceleration - Accelerometer Y')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(accel_z, marker='s', linestyle='-', label='Accel Z', color='green')
plt.xlabel('Sample')
plt.ylabel('m/s²')
plt.title('Acceleration - Accelerometer Z')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Plot rotation from the VN estimation in degrees on axes x, y, z (Fig 2)
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(rotation_x, marker='o', linestyle='-', label='Rotation X')
plt.xlabel('Sample')
plt.ylabel('Degrees')
plt.title('Rotation - Orientation X')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(rotation_y, marker='x', linestyle='-', label='Rotation Y', color='orange')
plt.xlabel('Sample')
plt.ylabel('Degrees')
plt.title('Rotation - Orientation Y')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(rotation_z, marker='s', linestyle='-', label='Rotation Z', color='green')
plt.xlabel('Sample')
plt.ylabel('Degrees')
plt.title('Rotation - Orientation Z')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Three 1D histograms of rotation in x, y, z (Fig 3)
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.hist(rotation_x, bins=30, color='blue', alpha=0.7)
plt.xlabel('Degrees')
plt.ylabel('Frequency')
plt.title('Histogram - Rotation X')

plt.subplot(3, 1, 2)
plt.hist(rotation_y, bins=30, color='orange', alpha=0.7)
plt.xlabel('Degrees')
plt.ylabel('Frequency')
plt.title('Histogram - Rotation Y')

plt.subplot(3, 1, 3)
plt.hist(rotation_z, bins=30, color='green', alpha=0.7)
plt.xlabel('Degrees')
plt.ylabel('Frequency')
plt.title('Histogram - Rotation Z')
plt.tight_layout()
plt.show()

def extract_gyro_from_string(data_string):
    """Extracts gyro values using regex to handle non-standard characters."""
    # Use regex to find floating-point numbers in the string
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', data_string)
    try:
        # Gyro values are assumed to be the 4th, 5th, and 6th numbers in the string
        gyro_x = float(matches[3])
        gyro_y = float(matches[4])
        gyro_z = float(matches[5])
        return gyro_x, gyro_y, gyro_z
    except (IndexError, ValueError) as e:
        print(f"Error parsing data string: {e}")
        return None, None, None

def read_rosbag_gyro_data(bag_file, topic='/vectornav'):
    """Reads gyro data from the specified ROSbag and topic."""
    gyro_x, gyro_y, gyro_z = [], [], []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            try:
                gx, gy, gz = extract_gyro_from_string(msg.data)
                if gx is not None and gy is not None and gz is not None:
                    gyro_x.append(gx)
                    gyro_y.append(gy)
                    gyro_z.append(gz)
            except AttributeError as e:
                print(f"Error reading message: {e}")
                continue

    return np.array(gyro_x), np.array(gyro_y), np.array(gyro_z)

def calculate_allan_variance(data, axis_label):
    """Calculates Allan variance and extracts noise parameters."""
    # Call oadev and capture all return values flexibly
    result = at.oadev(data, rate=1.0, taus='all')

    # Extract taus and oadev from result, regardless of its length
    if isinstance(result, tuple):
        taus = result[0]
        oadev = result[1]
    else:
        raise ValueError("Unexpected return type from oadev.")

    # Plot Allan variance
    plt.figure(figsize=(8, 6))
    plt.loglog(taus, oadev, label=f'Allan Deviation - Gyro {axis_label}')
    plt.xlabel('Tau (s)')
    plt.ylabel('Allan Deviation')
    plt.title(f'Allan Variance for Gyro {axis_label}')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
    
    # Noise Parameter Calculation
    min_idx = np.argmin(oadev)
    B = oadev[min_idx]  # Bias Instability
    tau_1_sec_index = np.where(taus == 1.0)[0]
    N = oadev[tau_1_sec_index[0]] if len(tau_1_sec_index) > 0 else "Not available"
    K_slope = np.polyfit(np.log10(taus[min_idx:]), np.log10(oadev[min_idx:]), 1)[0]
    K = 10**K_slope  # Rate Random Walk
    
    # Print results
    print(f"Noise parameters for Gyro {axis_label}:")
    print(f"  Bias Instability (B): {B}")
    print(f"  Angle Random Walk (N): {N}")
    print(f"  Rate Random Walk (K): {K}\n")
    
    return B, N, K

def main():
    bag_file_path = '/home/sha/Docs/northeastern/RSN/LAB4/Analysis/LocationD.bag'  # Update to your ROSbag file path
    gyro_x, gyro_y, gyro_z = read_rosbag_gyro_data(bag_file_path, topic='/vectornav')
    if gyro_x.size == 0 or gyro_y.size == 0 or gyro_z.size == 0:
        print("Error: No data extracted from ROSbag.")
        return
    # Calculate Allan variance and noise parameters for each gyro axis
    B_x, N_x, K_x = calculate_allan_variance(gyro_x, 'X')
    B_y, N_y, K_y = calculate_allan_variance(gyro_y, 'Y')
    B_z, N_z, K_z = calculate_allan_variance(gyro_z, 'Z')

main()
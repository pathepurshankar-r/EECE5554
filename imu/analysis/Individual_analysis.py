import rosbag
import numpy as np
import matplotlib.pyplot as plt
import allantools as at
import re

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

if __name__ == "__main__":
    main()

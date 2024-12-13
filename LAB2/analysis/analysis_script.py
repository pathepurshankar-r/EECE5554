import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to parse GPS data from text files
def parse_gps_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("$GNGGA"):
                parts = line.split(',')
                if len(parts) > 9:
                    try:
                        latitude = float(parts[2]) / 100.0
                        longitude = float(parts[4]) / 100.0
                        altitude = float(parts[9])
                        fix_quality = int(parts[6])
                        data.append([latitude, longitude, altitude, fix_quality])
                    except ValueError:
                        continue
    return pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Altitude', 'FixQuality'])

# Load data from text files
stationary_open = parse_gps_data('/home/sha/Docs/northeastern/RSN/EECE5554/LAB2/data/stationary-open.txt')
stationary_occluded = parse_gps_data('/home/sha/Docs/northeastern/RSN/EECE5554/LAB2/data/occluded-stationary.txt')
moving_open = parse_gps_data('/home/sha/Docs/northeastern/RSN/EECE5554/LAB2/data/moving-open.txt')
moving_occluded = parse_gps_data('/home/sha/Docs/northeastern/RSN/EECE5554/LAB2/data/occluded-moving.txt')

# Generate UTM coordinates (Easting and Northing)
# Simplified here for demonstration purposes
stationary_open['Easting'] = stationary_open['Longitude']
stationary_open['Northing'] = stationary_open['Latitude']

stationary_occluded['Easting'] = stationary_occluded['Longitude']
stationary_occluded['Northing'] = stationary_occluded['Latitude']

moving_open['Easting'] = moving_open['Longitude']
moving_open['Northing'] = moving_open['Latitude']

# Plot UTM Easting vs Northing
plt.figure()
plt.scatter(stationary_open['Easting'], stationary_open['Northing'], label='Open', c='blue')
plt.scatter(stationary_occluded['Easting'], stationary_occluded['Northing'], label='Occluded', c='red')
plt.xlabel('UTM Easting (m)')
plt.ylabel('UTM Northing (m)')
plt.legend()
plt.title('UTM Easting vs Northing')
plt.savefig('utm_easting_northing.png')

# Plot Altitude vs Time
plt.figure()
time_open = range(len(stationary_open))
time_occluded = range(len(stationary_occluded))
plt.plot(time_open, stationary_open['Altitude'], label='Open', c='blue')
plt.plot(time_occluded, stationary_occluded['Altitude'], label='Occluded', c='red')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.title('Altitude vs Time')
plt.savefig('altitude_time.png')

# Calculate deviation from centroid
centroid_open = stationary_open[['Easting', 'Northing']].mean()
centroid_occluded = stationary_occluded[['Easting', 'Northing']].mean()

stationary_open['Deviation'] = np.sqrt(
    (stationary_open['Easting'] - centroid_open['Easting'])**2 +
    (stationary_open['Northing'] - centroid_open['Northing'])**2
)
stationary_occluded['Deviation'] = np.sqrt(
    (stationary_occluded['Easting'] - centroid_occluded['Easting'])**2 +
    (stationary_occluded['Northing'] - centroid_occluded['Northing'])**2
)

# Histogram of deviations
plt.figure()
plt.hist(stationary_open['Deviation'], bins=30, label='Open', alpha=0.7)
plt.hist(stationary_occluded['Deviation'], bins=30, label='Occluded', alpha=0.7)
plt.xlabel('Deviation from Centroid (m)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of Deviation from Centroid')
plt.savefig('deviation_histogram.png')

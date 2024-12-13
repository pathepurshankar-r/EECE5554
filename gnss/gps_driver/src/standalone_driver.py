#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import serial
import utm
from gps_driver.msg import Customgps
from std_msgs.msg import Header
from datetime import datetime, timezone, timedelta
import time
import math

#Conversion of Latitude or Longitude from DDmm.mm to DD.dddd 
def degmin_to_degdec(x):

    degrees = int(x / 100)
    minutes = x - degrees * 100
    decimal_degrees = degrees + minutes / 60
    return decimal_degrees

#Conversion of timestamp in hhmmss.sss format to rospy.Time
from datetime import datetime, timedelta

def convert_timestamp(timestamp_str):

    if not timestamp_str:
        return rospy.Time.now()
    try:
        timestamp_float = float(timestamp_str)
        hours = int(timestamp_float // 10000)
        minutes = int((timestamp_float % 10000) // 100)
        seconds = timestamp_float % 100
        if seconds >= 60.0:
            seconds -= 60.0
            minutes += 1
        if minutes >= 60:
            minutes -= 60
            hours += 1
        if hours >= 24:
            hours -= 24
        now = datetime.utcnow()
        gps_datetime = datetime(
            year=now.year,
            month=now.month,
            day=now.day,
            hour=hours,
            minute=minutes,
            second=int(seconds),
            microsecond=int((seconds - int(seconds)) * 1e6)
        )
        if gps_datetime > now:
            gps_datetime -= timedelta(days=1)
        ros_time = rospy.Time.from_sec(gps_datetime.timestamp())
        return ros_time

    except ValueError as e:
        rospy.logerr(f"Invalid timestamp: {e}")
        return rospy.Time.now()

if __name__ == '__main__':
    SENSOR_NAME = "GPS"
    rospy.init_node('gps_talker', anonymous=True)
    serial_port_path = '/dev/pts/14'
    baud_rate = 4800
    timeout_sec = 5.0
    try: 
        serial_port = serial.Serial(serial_port_path, baud_rate, timeout=timeout_sec)
    except serial.SerialException as e:
        rospy.logerr(f"Could not open serial port {serial_port_path}: {e}")
        exit(1)
    pub = rospy.Publisher('GPS_Data', Customgps, queue_size=10)
    rate = rospy.Rate(10)        
    try:
        while not rospy.is_shutdown():
            data_bytes = serial_port.readline()
            if not data_bytes:
                rospy.logwarn("No data received from GPS")
                continue

            data_str = data_bytes.decode('ascii', errors='ignore').strip()
            if not data_str.startswith('$GPGGA'):
                continue
            gpgga_split = data_str.split(',')

            try:
                timestamp_str = gpgga_split[1]
                lat_raw = float(gpgga_split[2]) if gpgga_split[2] else 0.0
                lat_dir = gpgga_split[3]
                lon_raw = float(gpgga_split[4]) if gpgga_split[4] else 0.0
                lon_dir = gpgga_split[5]
                altitude = float(gpgga_split[9]) if gpgga_split[9] else 0.0
                gps_time = convert_timestamp(timestamp_str)
                latitude = degmin_to_degdec(lat_raw)
                longitude = degmin_to_degdec(lon_raw)
                if lat_dir == 'S':
                    latitude = -latitude
                if lon_dir == 'W':
                    longitude = -longitude
                easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)
                utm_msg = Customgps()
                utm_msg.header = Header()
                utm_msg.header.frame_id = 'GPS1_Frame'
                utm_msg.header.stamp = gps_time
                utm_msg.latitude = latitude
                utm_msg.longitude = longitude
                utm_msg.altitude = altitude
                utm_msg.utm_easting = round(easting, 2)
                utm_msg.utm_northing = round(northing, 2)
                utm_msg.zone = zone_number
                utm_msg.letter = zone_letter
                utm_msg.hdop = gpgga_split[8]
                utm_msg.gpgga_read = data_str
                print("GPS Data:")
                print(f"Latitude: {utm_msg.latitude}")
                print(f"Longitude: {utm_msg.longitude}")
                print(f"Altitude: {utm_msg.altitude} m")
                print(f"UTM Easting: {utm_msg.utm_easting} m")
                print(f"UTM Northing: {utm_msg.utm_northing} m")
                print(f"Zone: {utm_msg.zone}")
                print(f"Letter: {utm_msg.letter}")
                print(f"HDOP: {utm_msg.hdop}")
                print(f"GPGGA Read: {utm_msg.gpgga_read}")
                
                pub.publish(utm_msg)
                rospy.loginfo(f"Published GPS data at timestamp: {gps_time.to_sec()}")

            except ValueError as e:
                rospy.logerr(f"Error parsing GPS data: {e}")
                continue

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        serial_port.close()
        rospy.loginfo("Shutting down GPS node")
#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def talker():
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    
    while not rospy.is_shutdown():
        text_str = "Sensing and Navigation %s" % rospy.get_time()
        rospy.loginfo(text_str)
        pub.publish(text_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

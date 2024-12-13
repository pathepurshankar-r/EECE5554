#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(data):

    msg_mod = data.data
    
    if len(msg_mod) > 1:
    	msg_mod = msg_mod[1] + msg_mod[0] + msg_mod[2:]
    
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', msg_mod)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', String, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

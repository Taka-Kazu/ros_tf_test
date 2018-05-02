#!/usr/bin/env python

import rospy
from ros_tf_test import RosTFTest

if __name__ == '__main__':
  try:
    rtt = RosTFTest()
    rtt.train(1000)
  except rospy.ROSInterruptException: pass



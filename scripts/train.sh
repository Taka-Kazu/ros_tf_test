#!/bin/bash

gnome-terminal -e "/opt/ros/kinetic/bin/roscore" --geometry=45x12+0+0 & sleep 1s

gnome-terminal -e "/opt/ros/kinetic/bin/rosrun ros_tf_test learn_and_save.py" --geometry=45x12+485+0 & sleep 1s

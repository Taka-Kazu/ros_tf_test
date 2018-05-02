#!/bin/sh

gnome-terminal -e "/opt/ros/kinetic/bin/roscore" --geometry=45x12+0+0 & sleep 1s

gnome-terminal -e "/opt/ros/kinetic/bin/rosparam set usb_cam/video_device '/dev/video0'" --geometry=45x12+480+0 & sleep 1s
gnome-terminal -e "/opt/ros/kinetic/bin/rosrun usb_cam usb_cam_node" --geometry=45x12+480+0 & sleep 1s

gnome-terminal -e "/opt/ros/kinetic/bin/rosrun image_view image_view image:=/usb_cam/image_raw" --geometry=45x12+960+0

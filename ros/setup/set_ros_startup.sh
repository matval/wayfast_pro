#!/bin/sh
sudo cp env.sh /etc/ros/env.sh
sudo cp my_roscore.service /etc/systemd/system/my_roscore.service

sudo systemctl enable my_roscore.service

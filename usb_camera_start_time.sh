#!/bin/bash

# 获取USB相机设备路径
camera_path=$(udevadm info -q path -n /dev/video0)

# 获取设备创建时间
start_time=$(udevadm info -a -p $camera_path | grep -Eo 'ATTR{datecreated}=="[0-9\-:]+"' | sed 's/ATTR{datecreated}=="//;s/"//')

echo "USB camera start time: $start_time"

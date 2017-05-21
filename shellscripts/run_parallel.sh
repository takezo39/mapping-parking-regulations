#!/bin/bash
#for frame_i in  1 5001 10001 15001 20001 25001 30001 35001 40001 45001 50001 55001 60001 65001 70001 75001
for frame_i in  1 5001 
do
    frame_f=$((frame_i + 4999))
    echo "Running $frame_i to $frame_f"
    if [ $frame_i -eq 1 ]
    then
        nohup python sign_detection/features/make_dataset_and_build_features.py -run_code="170329" -gps_path="data/interim/170329/gps/3.29.12.47_35.pkl" -frame_i="$frame_i" -frame_f="$frame_f" -hough_radii_increment=4 -keep_output &
    else
        sleep 5
        nohup python sign_detection/features/make_dataset_and_build_features.py -run_code="170329" -gps_path="data/interim/170329/gps/3.29.12.47_35.pkl" -frame_i="$frame_i" -frame_f="$frame_f" -hough_radii_increment=4 -keep_output -dont_del_old &
    fi
done

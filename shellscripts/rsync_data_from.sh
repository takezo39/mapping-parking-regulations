#!/bin/bash

i="0"
while [ $i -lt 1 ]
do
    echo "syncing"
    rsync -r --include-from rsync_data_includes.txt $1:~/static-data-sign-detection/data/interim/170329 ../data/interim/
    sleep 600s
done

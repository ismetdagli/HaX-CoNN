#!/bin/sh

sudo tegrastats | stdbuf -oL  sed -n 's/.*EMC_FREQ \([0-9]\+%\).*/\1/p'  > /tmp/temp_times.txt &
/usr/src/tensorrt/bin/trtexec --iterations=100 --avgRuns=1 --warmUp=10000 --duration=0 --loadEngine=$1 > /dev/null
sudo tegrastats --stop
sort -nr /tmp/temp_times.txt | head -1


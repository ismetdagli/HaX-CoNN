#!/bin/sh

sudo pkill tegrastats
sudo tegrastats | stdbuf -oL  sed -n 's/.*EMC_FREQ \([0-9]\+%\).*/\1/p'  > "$2" &
/usr/src/tensorrt/bin/trtexec --iterations=100 --avgRuns=1 --warmUp=10000 --duration=0 --loadEngine="$1" > /dev/null
sudo tegrastats --stop || sudo pkill tegrastats
sort -nr $2 | head -1


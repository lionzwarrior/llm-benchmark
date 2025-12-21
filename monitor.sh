#!/bin/bash

# --- Start Monitoring Processes ---
echo "Starting monitoring..."

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu --format=csv -l 1 > gpu.csv &
PID_NVIDIA=$!

top -b -d 1 | awk '
/^%Cpu/     {print > "cpu.txt"; fflush("cpu.txt")}
/^MiB Mem/  {print > "mem.txt"; fflush("mem.txt")}
' &
PID_TOP=$!


# Store them in a file so the parent script can kill them
echo $PID_NVIDIA > monitor_pids.txt
echo $PID_TOP >> monitor_pids.txt

# Keep monitor.sh alive so it can be killed
wait

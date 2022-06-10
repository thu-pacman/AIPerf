ps -aux | grep root | grep train | grep -v grep | awk '{print $2}' | xargs kill
ps -aux | grep root | grep tmp.sh | grep -v grep | awk '{print $2}' | xargs kill
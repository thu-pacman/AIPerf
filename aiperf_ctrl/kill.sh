ps -aux | grep image | grep -v grep | awk '{print $2}' | xargs kill
ps -aux | grep tmp.sh | grep -v grep | awk '{print $2}' | xargs kill
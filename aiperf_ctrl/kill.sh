ps -aux | grep thu | grep python | grep -v grep | awk '{print $2}' | xargs kill
ps -aux | grep thu | grep tmp.sh | grep -v grep | awk '{print $2}' | xargs kill
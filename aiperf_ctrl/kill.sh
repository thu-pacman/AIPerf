ps -aux | grep wxp | grep python | grep -v grep | awk '{print $2}' | xargs kill
ps -aux | grep wxp | grep tmp.sh | grep -v grep | awk '{print $2}' | xargs kill
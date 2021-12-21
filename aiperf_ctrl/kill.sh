ps -aux | grep python | awk '{print $2}' | xargs kill
ps -aux | grep tmp.sh | awk '{print $2}' | xargs kill
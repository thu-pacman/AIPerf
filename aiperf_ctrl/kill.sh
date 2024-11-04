#!/bin/bash
kill_proc() {
  local process_name="$1"
  if [ -z "$process_name" ]; then
    echo "Please provide a process name."
    return 1
  fi
  for pid in $(pgrep -f "$process_name"); do
    echo "Killing process ID $pid containing '$process_name'"
    kill -9 $pid
  done
}

kill_proc "image"
kill_proc "tmp.sh"
kill_proc "multiprocessing.forkserver"

#!/bin/bash
BASE=/data/home/minimind-system
if [ -f "$BASE/service.pid" ]; then
  PID=$(cat "$BASE/service.pid")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" && echo "stopped $PID"
  else
    echo "not running ($PID)"
  fi
  rm -f "$BASE/service.pid"
else
  echo "no pid file"
fi

#!/bin/bash

echo "Starting server"
python3 ./server.py --fr_rate=1.0 --strategy='fedavg' &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 1`; do
  echo "Starting client $i"
  python3 ./client.py &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
read -p "Press [Enter] to exit..."

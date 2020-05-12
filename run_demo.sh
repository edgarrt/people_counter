#!/bin/bash

# Pull docker images
docker pull edgarrt/ppl_counter_mqtt_server
docker pull edgarrt/ppl_counter_ui
docker pull edgarrt/ppl_counter_py

# Run docker images
docker run -d -p 3001:3001 -p 3002:3002 edgarrt/ppl_counter_mqtt_server

docker run -d -p 3000:3000 edgarrt/ppl_counter_ui

docker run -d -p 3004:3004 edgarrt/ppl_counter_py ffserver -f ./ffmpeg/server.conf


docker ps --last 1 --format {{.ID}} > last_container.txt


start chrome http://localhost:3000

set /p ID=<last_container.txt

docker exec -it %ID% bash

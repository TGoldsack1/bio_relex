#!/bin/bash
# Author: Tomas Goldsack

nohup /home/tomas/models/public_mm/bin/skrmedpostctl start
nohup /home/tomas/models/public_mm/bin/wsdserverctl start
sleep 60
# nohup python create_background_knowledge.py
nohup python create_discourse_graph.py
/home/tomas/models/public_mm/bin/skrmedpostctl stop
/home/tomas/models/public_mm/bin/wsdserverctl stop

curl -X POST -H "Content-Type: application/json" -d '{"value1":"Graph creation terminated"}' https://maker.ifttt.com/trigger/notify/with/key/d-tdD7KN2shbNEa9HnejHG
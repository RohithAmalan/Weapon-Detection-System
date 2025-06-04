import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_address = s.getsockname()[0]
print(ip_address)

import paho.mqtt.client as mqtt
import requests
import json

def on_message(client, userdata, message):
    print("li")
    data1 =[]
    receivedstring = str(message.payload.decode("utf-8"))
    data1=receivedstring.split(",")
    print(data1)
    with open('config.json', 'w') as json_file:
        json.dump(data1, json_file)

broker_address="broker.hivemq.com"
client = mqtt.Client("PROJECT") 
client.connect(broker_address) 
client.on_message=on_message 
client.subscribe("WEAPON-NT")

client.loop_forever()
import random
import time

from paho.mqtt import client as mqtt_client


broker = "192.168.0.150"
port = 1883
topic = "eye_exam/send"
topic2 = "eye_exam/ack"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
received_msg = "none"

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set("ntust", "ntustee")
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def on_message(client, userdata, msg):
    global received_msg
    received_msg = msg.payload
    #print(msg.topic+" "+ msg.payload)

def run():
    global received_msg
    msg = "left"
    temp = b'turn left'
    while True:
        time.sleep(5)
        client.publish(topic, msg)
        client.subscribe(topic2)
        client.on_message = on_message
        while True:
            time.sleep(1.5)
            if received_msg == temp:
                received_msg = ""
                break
            print(received_msg)
            print(temp)
            client.publish(topic, msg)
        if msg == "left":
            msg = "right"
            temp = b'turn right'
        else:
            msg = "left"
            temp = b'turn left'


if __name__ == '__main__':
    client = connect_mqtt()
    client.loop_start()
    run()

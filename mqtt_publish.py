import random
import time

from paho.mqtt import client as mqtt_client


broker = "192.168.0.150"
port = 1883
topic = "test/firstTest"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'


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


def publish_mqtt():
    client = connect_mqtt()
    client.loop_start()
    while True:
        time.sleep(5)
        msg = "LEDON"
        client.publish(topic, msg)


if __name__ == '__main__':
    publish_mqtt()

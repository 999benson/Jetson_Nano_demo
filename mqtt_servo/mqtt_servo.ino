#include <WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

Servo myservo;

// Update these with values suitable for your network.
long serial_speed = 115200;
//const char* ssid = "dlink-D716";
//const char* password = "gog16888";
//const char* mqtt_server = "192.168.0.150";
const char* ssid = "Hsu__47";
const char* password = "cloud041313";
const char* mqtt_server = "172.20.10.2";
int mqtt_port = 1883;
const char* user_name = "ntust"; // 連接 MQTT broker 的帳號密碼
const char* user_password = "123";

// 訂閱的主題：收到 0 關閉 LED，1 打開LED
const char* topic_subscribe = "eye_exam/send"; 
const char* topic_publish = "eye_exam/ack";

const int LEDPin = 22;    // LED connected to digital pin 22

WiFiClient espClient;
PubSubClient client(espClient);
long lastMsg = 0;
char msg[50];

void setup_wifi() { // 連接Wifi
  delay(10);
  // We start by connecting to a WiFi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  randomSeed(micros());
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) 
{
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  String messageTemp;
  
  for (int i = 0; i < length; i++) {
//    Serial.print((char)payload[i]);
    messageTemp += (char)payload[i];
  }
  Serial.println(messageTemp);
  if(messageTemp == "ack"){
    Serial.println("print hi");
    client.publish("test/firstTest", "hi");
  }else if(messageTemp == "LEFT" || messageTemp == "left"){
//    Serial.println("LED on");
//    digitalWrite(LEDPin, HIGH);
    Serial.println("turn left");
    client.publish(topic_publish, "turn left");
    myservo.write(65);
  }else if(messageTemp == "RIGHT" || messageTemp == "right"){
//    Serial.println("LED off");
//    digitalWrite(LEDPin, LOW);
    Serial.println("turn right");
    client.publish(topic_publish, "turn right");
    myservo.write(115);
  }
} //end callback

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.println("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "espClient-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str(),user_name,user_password)) {
      Serial.println("connected");
      client.subscribe(topic_subscribe);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(serial_speed);
  pinMode(LEDPin, OUTPUT);
  myservo.attach(13);
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
    
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
//  digitalWrite(LEDPin, HIGH); // Set GPIO22 active high
//  delay(1000);  // delay of one second
//  digitalWrite(LEDPin, LOW); // Set GPIO22 active low
//  delay(1000);
//  long now = millis();
//  if (now - lastMsg > 5000) {
//    lastMsg = now;
//    
//    // Convert the value to a char array
////    char tempString[8];
////    dtostrf(temperature, 1, 2, tempString);
////    Serial.print("Temperature: ");
////    Serial.println(tempString);
////    client.publish("test/firstTest", tempString);
//    Serial.println("print hi");
//    client.publish("test/firstTest", "hi");
//  }
}

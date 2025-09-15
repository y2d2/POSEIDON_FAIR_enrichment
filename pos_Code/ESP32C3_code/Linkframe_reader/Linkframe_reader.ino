#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_NeoPixel.h>

// --- LED ---
#define LED_PIN 10    
#define NUM_LEDS 1 
Adafruit_NeoPixel pixels(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

// --- WiFi ----
const char* ssid = "MultiAgentSLAM2";
const char* password = "MultiAgentSLAM!";

const char* pc_ip = "192.168.0.100";  // Replace with your PC’s IP
const int pc_port = 5005;

WiFiUDP udp;

// --- UART SERIAL ---
#define RXD1 20   // ESP32-C3 RX
#define TXD1 21   // ESP32-C3 TX
static uint16_t pos = 0;
#define BUF_SIZE 512
uint8_t buf[BUF_SIZE];


void enableIMU(){ 
  
}

void setup() {
  Serial.begin(115200);  // debug via USB
  delay(1000);

  pixels.begin(); // initialize NeoPixel library
  pixels.setPixelColor(0, pixels.Color(255, 255, 255)); // R,G,B
  pixels.show();
  // Start UART to LinkTrack
  Serial1.begin(921600, SERIAL_8N1, RXD1, TXD1);

  // Connect WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected! IP=" + WiFi.localIP().toString());
  pixels.setPixelColor(0, pixels.Color(0, 255, 0)); // R,G,B
  pixels.show();
}


void dumpRawFrame(const uint8_t *buf, int len) {
  if (buf[0]==0x55 && buf[1]==0x04) { 
    Serial.print("Raw frame: ");
    for (size_t i = 0; i <= len; i++) {
      if (buf[i] < 0x10) Serial.print("0");  // leading zero for 1-digit hex
      Serial.print(buf[i], HEX);
      Serial.print(" ");
    }
    Serial.println();
  // }
  } else {
    Serial.print("Invalid frame");
  }
}


void parseFrame(const uint8_t *buf) {
  // Offsets from your table
  // Serial.println("Test");  //
  String message = "";
  uint8_t id = buf[5];
  message = message + String(id)+";";
  // IMUData imu;
  uint32_t systime;
  memcpy(&systime, &buf[6],4);
  message += String(systime)+";";
  float gx, gy, gz, ax, ay, az;
  memcpy(&gx, &buf[40], 4);
  memcpy(&gy, &buf[44], 4);
  memcpy(&gz, &buf[48], 4);
  memcpy(&ax, &buf[52], 4);
  memcpy(&ay, &buf[56], 4);
  memcpy(&az, &buf[60], 4);

  message += String(gx) +";" + String(gy) +";"  + String(gz)+ ";";
  message += String(ax) +";" + String(ay) +";"  + String(az)+ ";";

  // Serial.printf("Gyro: %f %f %f rad/s\n", imu.gx, imu.gy, imu.gz);
  // Serial.printf("Accel: %f %f %f m/s^2\n", imu.ax, imu.ay, imu.az);

  // Valid node count
  uint8_t validCount = buf[118];
  message += String(validCount) +";";
  // Serial.printf("Valid ranges: %d\n", validCount);

  int offset = 119;
  for (int i = 0; i < validCount; i++) {
    // RangeData rd;
    uint8_t idr;
    float distance; // meters
    int8_t fp_rssi;
    int8_t rx_rssi;
    idr = buf[offset + 1];
    int32_t distRaw = (buf[offset + 2]) | (buf[offset + 3] << 8) | (buf[offset + 4] << 16);
    if (distRaw & 0x800000) distRaw |= 0xFF000000; // sign extend
    distance = distRaw / 1000.0f;
    fp_rssi = buf[offset + 5] / -2;
    rx_rssi = buf[offset + 6] / -2;

    message += String(idr) +";" + String(distance) +";" + String(fp_rssi) + ";" +String(rx_rssi)+ ";";
    // Serial.printf("Node %d (role %d): %.2f m, FP_RSSI=%d dB, RX_RSSI=%d dB\n",
    //               rd.id, rd.role, rd.distance, rd.fp_rssi, rd.rx_rssi);

    offset += 13; // each block length
  }
  Serial.println(message);
  udp.beginPacket(pc_ip, pc_port);
  udp.print(message);
  udp.endPacket();
}

bool checkSum(const uint8_t *buf, int len) {
   // checksum: sum of all previous bytes
  uint8_t sum = 0;
  for (int i = 0; i <= len - 3; i++) sum += buf[i];
  if (sum != buf[len - 2]) {
    Serial.println("Checksum error");
    return false;
  } 
  return true;
}

void loop() {

  while (Serial1.available()) {
    buf[pos] = Serial1.read(); 
    if (buf[pos] == 0x04) { 
      if (pos > 1) { 
        if (buf[pos-1] == 0x55){ 
          // dumpRawFrame(buf, pos);
          if (checkSum(buf, pos)){
            pixels.setPixelColor(0, pixels.Color(0, 100, 0)); // R,G,B
            pixels.show();
            parseFrame(buf);
          }
          else {
            pixels.setPixelColor(0, pixels.Color(255, 0, 0)); // R,G,B
            pixels.show();
          }
          buf[0] = 0x55;
          buf[1] = 0x04;
          pos = 1;
          pixels.setPixelColor(0, pixels.Color(0, 0, 0)); // R,G,B
          pixels.show();
        }
      }
    }
    pos++;
  }    
    


  // while (Serial1.available()) {
  //   uint8_t b = Serial1.read();
  //   Serial.print(b, HEX);   // print as hex
  //   Serial.print(" ");
  // }
  
  // if (readFrame2()) {
  //   // dumpRawFrame(buf);
  //   parseFrame();

    // // Parse minimal NodeFrame2
    // uint16_t id = buf[0] | (buf[1] << 8);
    // uint8_t role = buf[2];

    // int16_t acc[3];
    // int16_t gyro[3];
    // for (int i = 0; i < 3; i++) {
    //   acc[i] = (int16_t)(buf[3 + i*2] | (buf[4 + i*2] << 8));
    // }
    // for (int i = 0; i < 3; i++) {
    //   gyro[i] = (int16_t)(buf[9 + i*2] | (buf[10 + i*2] << 8));
    // }

    // uint8_t rangeCount = buf[15];
    // int offset = 16;

    // String message = "ID=" + String(id) +
    //                  " ACC=" + String(acc[0]) + "," + String(acc[1]) + "," + String(acc[2]) +
    //                  " GYRO=" + String(gyro[0]) + "," + String(gyro[1]) + "," + String(gyro[2]) +
    //                  " Ranges=";

    // for (int i = 0; i < rangeCount; i++) {
    //   uint16_t rid = buf[offset] | (buf[offset+1] << 8);
    //   uint32_t dist = buf[offset+2] | (buf[offset+3] << 8) | (buf[offset+4] << 16) | (buf[offset+5] << 24);
    //   offset += 6;

    //   message += "[" + String(rid) + ":" + String(dist) + "mm] ";
    // }

    // Serial.println(message);

    // // Send over UDP

  // }
}
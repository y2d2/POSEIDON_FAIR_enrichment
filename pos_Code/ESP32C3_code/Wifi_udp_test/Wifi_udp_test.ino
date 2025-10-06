#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "Olympus";
const char* password = "Z3us&H3ra!";

const char* pc_ip = "192.168.0.134";  // Replace with your PC’s IP
const int pc_port = 5005;

WiFiUDP udp;

// ---- UART pins ----
#define RXD1 20   // ESP32-C3 RX
#define TXD1 21   // ESP32-C3 TX

// ---- Buffer for incoming frame ----
#define BUF_SIZE 256
uint8_t buf[BUF_SIZE];


void setup() {
  Serial.begin(115200);  // debug via USB
  delay(1000);

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
}

bool readFrame() {
  // Wait for header 0x55 0x02
  if (Serial1.available() >= 2) {
    if (Serial1.peek() == 0x55) {
      Serial1.read();
      if (Serial1.peek() == 0x04) {
        Serial1.read();
        // Now read length
        while (Serial1.available() < 2) return false;
        uint16_t len = Serial1.read() | (Serial1.read() << 8);

        if (len > BUF_SIZE) return false;
        while (Serial1.available() < len) return false;

        // Read payload
        for (int i = 0; i < len; i++) {
          buf[i] = Serial1.read();
        }

        return true;
      }
    } else {
      Serial1.read(); // discard bad byte
    }
  }
  return false;
}

void loop() {
  // while (Serial1.available()) {
  //   uint8_t b = Serial1.read();
  //   Serial.print(b, HEX);   // print as hex
  //   Serial.print(" ");
  // }

  if (readFrame()) {
    // Parse minimal NodeFrame2
    uint16_t id = buf[0] | (buf[1] << 8);
    uint8_t role = buf[2];

    int16_t acc[3];
    int16_t gyro[3];
    for (int i = 0; i < 3; i++) {
      acc[i] = (int16_t)(buf[3 + i*2] | (buf[4 + i*2] << 8));
    }
    for (int i = 0; i < 3; i++) {
      gyro[i] = (int16_t)(buf[9 + i*2] | (buf[10 + i*2] << 8));
    }

    uint8_t rangeCount = buf[15];
    int offset = 16;

    String message = "ID=" + String(id) +
                     " ACC=" + String(acc[0]) + "," + String(acc[1]) + "," + String(acc[2]) +
                     " GYRO=" + String(gyro[0]) + "," + String(gyro[1]) + "," + String(gyro[2]) +
                     " Ranges=";

    for (int i = 0; i < rangeCount; i++) {
      uint16_t rid = buf[offset] | (buf[offset+1] << 8);
      uint32_t dist = buf[offset+2] | (buf[offset+3] << 8) | (buf[offset+4] << 16) | (buf[offset+5] << 24);
      offset += 6;

      message += "[" + String(rid) + ":" + String(dist) + "mm] ";
    }

    Serial.println(message);

    // Send over UDP
    udp.beginPacket(pc_ip, pc_port);
    udp.print(message);
    udp.endPacket();
  }
}
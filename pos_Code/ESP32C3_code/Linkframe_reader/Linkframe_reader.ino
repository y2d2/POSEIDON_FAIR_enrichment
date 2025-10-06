#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_NeoPixel.h>
#include <Wire.h>        // I²C bus
#include "ICM_20948.h"   // SparkFun library

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

// --- IMU ---
#define WIRE_PORT Wire 
ICM_20948_I2C myICM;
bool imu_on = false;

bool setupIMU(){ 
  WIRE_PORT.begin(3, 2);
  WIRE_PORT.setClock(400000);
  bool initialized = false;
  int i = 0; 
  while (i < 10)
  {
    i++;
    myICM.begin(WIRE_PORT, 0x68);

    Serial.print(F("Initialization of the sensor returned: "));
    Serial.println(myICM.statusString());
    if (myICM.status != ICM_20948_Stat_Ok)
    {
      Serial.println("Trying again...");
      pixels.setPixelColor(0, pixels.Color(100, 0, 0)); // R,G,B
      pixels.show();

      delay(50);
    }
    else
    {
      initialized = true;
      break;
    }
  }
  if (!initialized) { 
    return false;
  }

  Serial.println("ICM-20948 ready!");
  
  pixels.setPixelColor(0, pixels.Color(0, 0, 100)); // R,G,B
  pixels.show();

  myICM.swReset();
  if (myICM.status != ICM_20948_Stat_Ok)
  {
    Serial.print(F("Software Reset returned: "));
    Serial.println(myICM.statusString());
  }
  delay(250);

  // Now wake the sensor up
  myICM.sleep(false);
  myICM.lowPower(false);

  // The next few configuration functions accept a bit-mask of sensors for which the settings should be applied.

  // Set Gyro and Accelerometer to a particular sample mode
  // options: ICM_20948_Sample_Mode_Continuous
  //          ICM_20948_Sample_Mode_Cycled
  myICM.setSampleMode((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), ICM_20948_Sample_Mode_Continuous);
  if (myICM.status != ICM_20948_Stat_Ok)
  {
    Serial.print(F("setSampleMode returned: "));
    Serial.println(myICM.statusString());
  }
  
  // Set full scale ranges for both acc and gyr
  ICM_20948_fss_t myFSS; // This uses a "Full Scale Settings" structure that can contain values for all configurable sensors

  myFSS.a = gpm2; // (ICM_20948_ACCEL_CONFIG_FS_SEL_e)
                  // gpm2
                  // gpm4
                  // gpm8
                  // gpm16

  myFSS.g = dps250; // (ICM_20948_GYRO_CONFIG_1_FS_SEL_e)
                    // dps250
                    // dps500
                    // dps1000
                    // dps2000

  myICM.setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), myFSS);
  if (myICM.status != ICM_20948_Stat_Ok)
  {
    Serial.print(F("setFullScale returned: "));
    Serial.println(myICM.statusString());
  }

  // Set up Digital Low-Pass Filter configuration
  ICM_20948_dlpcfg_t myDLPcfg;    // Similar to FSS, this uses a configuration structure for the desired sensors
  myDLPcfg.a = acc_d11bw5_n17bw;
                                  // acc_d473bw_n499bw; // (ICM_20948_ACCEL_CONFIG_DLPCFG_e)
                                  // acc_d246bw_n265bw      - means 3db bandwidth is 246 hz and nyquist bandwidth is 265 hz
                                  // acc_d111bw4_n136bw
                                  // acc_d50bw4_n68bw8
                                  // acc_d23bw9_n34bw4
                                  // acc_d11bw5_n17bw
                                  // acc_d5bw7_n8bw3        - means 3 db bandwidth is 5.7 hz and nyquist bandwidth is 8.3 hz
                                  // acc_d473bw_n499bw

  myDLPcfg.g = gyr_d11bw6_n17bw8; // (ICM_20948_GYRO_CONFIG_1_DLPCFG_e)
                                  // gyr_d361bw4_n376bw5
                                  // gyr_d196bw6_n229bw8
                                  // gyr_d151bw8_n187bw6
                                  // gyr_d119bw5_n154bw3
                                  // gyr_d51bw2_n73bw3
                                  // gyr_d23bw9_n35bw9
                                  // gyr_d11bw6_n17bw8
                                  // gyr_d5bw7_n8bw9
                                  // gyr_d361bw4_n376bw5

  myICM.setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), myDLPcfg);
  if (myICM.status != ICM_20948_Stat_Ok)
  {
    Serial.print(F("setDLPcfg returned: "));
    Serial.println(myICM.statusString());
  }

  // Choose whether or not to use DLPF
  // Here we're also showing another way to access the status values, and that it is OK to supply individual sensor masks to these functions
  ICM_20948_Status_e accDLPEnableStat = myICM.enableDLPF(ICM_20948_Internal_Acc, false);
  ICM_20948_Status_e gyrDLPEnableStat = myICM.enableDLPF(ICM_20948_Internal_Gyr, false);
  Serial.print(F("Enable DLPF for Accelerometer returned: "));
  Serial.println(myICM.statusString(accDLPEnableStat));
  Serial.print(F("Enable DLPF for Gyroscope returned: "));
  Serial.println(myICM.statusString(gyrDLPEnableStat));

  // Choose whether or not to start the magnetometer
  myICM.startupMagnetometer();
  if (myICM.status != ICM_20948_Stat_Ok)
  {
    Serial.print(F("startupMagnetometer returned: "));
    Serial.println(myICM.statusString());
  }

  Serial.println();
  Serial.println(F("Configuration complete!"));
  pixels.setPixelColor(0, pixels.Color(0, 100, 0)); // R,G,B
  pixels.show();
  return true;
}


void setup() {
  Serial.begin(115200);  // debug via USB
  delay(1000);

  pixels.begin(); // initialize NeoPixel library
  pixels.setPixelColor(0, pixels.Color(100, 100, 100)); // R,G,B
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
  pixels.setPixelColor(0, pixels.Color(0, 255, 255)); // R,G,B
  pixels.show();

  imu_on = setupIMU();
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
  // float gx, gy, gz, ax, ay, az;
  // memcpy(&gx, &buf[40], 4);
  // memcpy(&gy, &buf[44], 4);
  // memcpy(&gz, &buf[48], 4);
  // memcpy(&ax, &buf[52], 4);
  // memcpy(&ay, &buf[56], 4);
  // memcpy(&az, &buf[60], 4);

  // float q0, q1, q2, q3 =0;
  float ax =0, ay=0, az=0, gx =0, gy=0, gz=0, mx=0, my=0, mz=0, t=0;
  if (imu_on and myICM.dataReady()) {
    myICM.getAGMT();
    ax = myICM.accX(), ay = myICM.accY(), az = myICM.accZ();
    gx = myICM.gyrX(), gy = myICM.gyrY(), gz = myICM.gyrZ();
    mx = myICM.magX(), my = myICM.magY(), mz = myICM.magZ();
    t = myICM.temp();
  } else {
    pixels.setPixelColor(0, pixels.Color(255, 0, 0)); // R,G,B
    pixels.show();
  }
  message += String(mx) + ";" + String(my) + ";" +String(mz) + ";" +String(t) + ";";
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
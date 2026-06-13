#include <Servo.h>

Servo servoTrai; // ID 1 (Gạt trái)
Servo servoPhai; // ID 2 (Gạt phải)

const int PIN_TRAI = 9;
const int PIN_PHAI = 10;

const int TRAI_NGHI = 180;
const int TRAI_GAT = 110;
const int PHAI_NGHI = 10;
const int PHAI_GAT = 80;

const int THOI_GIAN_GIU = 2000;
const int Thoi_gian = 4000;

void setup() {
  Serial.begin(9600);
  servoTrai.attach(PIN_TRAI);
  servoPhai.attach(PIN_PHAI);
  servoTrai.write(TRAI_NGHI);
  servoPhai.write(PHAI_NGHI);
}

void loop() {
  if (Serial.available() > 0) {
    char lenh = Serial.read();

    if (lenh == '1') {
      servoTrai.write(TRAI_GAT);
      delay(THOI_GIAN_GIU);
      servoTrai.write(TRAI_NGHI);
    } else if (lenh == '2') {
      servoPhai.write(PHAI_GAT);
      delay(Thoi_gian);
      servoPhai.write(PHAI_NGHI);
    }
  }
}

# Smart Orange Classification System

## 1. Project overview
This project builds a vision-based orange sorting system using a Raspberry Pi camera, a laptop running YOLOv8/OpenCV, and an Arduino-controlled servo actuator. The current implementation focuses on detecting defective oranges (especially dark spots/marks) and triggering left/right sorting commands.

## 2. System workflow
1. Raspberry Pi captures video from the camera and streams it over HTTP.
2. The laptop receives the stream, runs YOLOv8 + OpenCV, and detects oranges.
3. The laptop sends `/control?id=1` or `/control?id=2` back to the Pi.
4. The Pi forwards the command through Serial to Arduino.
5. Arduino drives the servo to sort the fruit into the corresponding lane.

## 3. Technology stack
- Python + OpenCV
- YOLOv8 (Ultralytics)
- Flask on Raspberry Pi
- pyserial / UART Serial
- Arduino + Servo

## 4. Project structure
- src/laptop/ : main processing code on the laptop
- src/raspberry_pi/ : Pi streaming and control code
- src/arduino/ : Arduino firmware
- models/ : model weights directory
- docs/ : operation notes

## 5. Raspberry Pi setup
1. Install OS packages:
   sudo apt update
   sudo apt install -y python3-picamera2 python3-opencv python3-flask python3-serial
2. Install Python dependencies:
   pip install -r requirements-pi.txt
3. Start the stream server:
   python src/raspberry_pi/pi_stream_control.py

## 6. Laptop setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements-laptop.txt
   ```
2. Download `yolov8n.pt` and place it at `models/yolov8n.pt` before running the laptop script. Model weight files (`*.pt`) are ignored by Git, so they are not included when cloning the repository.
3. Set the Raspberry Pi IP address with the `PI_IP` environment variable if it is different from the default in the script.
   Windows PowerShell:
   ```powershell
   $env:PI_IP="192.168.1.20"
   ```
   macOS/Linux:
   ```bash
   export PI_IP=192.168.1.20
   ```
4. Run the main processing script:
   ```bash
   python src/laptop/main_detect_sort.py
   ```

## 7. Arduino upload
- Open src/arduino/servo_sorter.ino in Arduino IDE.
- Select the correct board and COM port.
- Upload the firmware to the Arduino.

## 8. Current sorting behavior
- The system detects oranges and marks defective ones using stain/spot analysis.
- Sorting commands are triggered based on the fruit position (left/right lane).
- The current version is designed for two sorting lanes, which matches the existing Arduino command set.

## 9. Notes
- Confirm the Raspberry Pi IP address before running the laptop code. You can override it with the `PI_IP` environment variable.
- Lighting and camera positioning strongly affect detection quality.
- Servo angle and delay may need adjustment depending on the hardware setup.

## 10. Future improvements
- Improve the detection and classification logic.
- Add a trained model for orange quality grading.
- Optimize logging and monitoring for deployment use.

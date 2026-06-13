from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2
import cv2
import serial
import time
import os

app = Flask(__name__)

# ================== CONFIG ==================

SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyACM0")
BAUD_RATE = 9600

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

# ================== ARDUINO SERIAL ==================

arduino = None

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[OK] Connected to Arduino at {SERIAL_PORT}")
except Exception as e:
    print(f"[WARN] Cannot connect to Arduino: {e}")
    print("[WARN] Video stream will still run, but control will be disabled.")

# ================== CAMERA ==================

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(
    main={
        "size": (FRAME_WIDTH, FRAME_HEIGHT),
        "format": "BGR888"
    }
)
picam2.configure(camera_config)
picam2.start()

time.sleep(1)
print("[OK] Camera started")

# ================== VIDEO STREAM ==================


def generate_frames():
    while True:
        frame = picam2.capture_array()

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return "Raspberry Pi video stream and control server is running."


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/control")
def control():
    servo_id = request.args.get("id")

    if servo_id not in ["1", "2"]:
        return jsonify({
            "status": "error",
            "message": "Invalid servo id. Use id=1 or id=2."
        }), 400

    if arduino is None:
        return jsonify({
            "status": "error",
            "message": "Arduino is not connected."
        }), 500

    try:
        arduino.write(servo_id.encode())
        print(f"[CONTROL] Sent command to Arduino: {servo_id}")

        return jsonify({
            "status": "ok",
            "sent": servo_id
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    print("[OK] Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)

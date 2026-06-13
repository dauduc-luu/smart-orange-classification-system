import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import requests

# --- CẤU HÌNH ---
PI_IP = os.getenv("PI_IP", "10.29.211.93")  # Có thể đổi bằng biến môi trường PI_IP
URL_VIDEO = f"http://{PI_IP}:5000/video"
URL_CONTROL = f"http://{PI_IP}:5000/control"

MODEL_PATH = Path("models/yolov8n.pt")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Please download yolov8n.pt and place it in the models/ folder."
    )
CONF_THRESHOLD = 0.5 

# BỎ biến MIN_FRUIT_SIZE_TO_CHECK để nó soi từ lúc quả còn xa

# Cấu hình vết mực
MIN_BLOB_AREA = 300 
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 60]) 

# Delay gạt (Cần căn chỉnh vì giờ phát hiện từ xa, thời gian trôi đến tay gạt sẽ lâu hơn)
DELAY_GAT = 2.0 

# --- BỘ NHỚ TRẠNG THÁI (QUAN TRỌNG) ---
# Lưu trạng thái của từng ID: {id: {'is_bad': False, 'max_error': 0}}
fruit_memory = defaultdict(lambda: {'is_bad': False, 'max_error': 0})
# Lưu ID đã gửi lệnh gạt để tránh cùng một quả bị gạt nhiều lần.
sent_ids = set()

# --- CLASS CAMERA ---
class VideoReceiver:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        while not self.stopped:
            if not self.stream.isOpened(): continue
            grabbed = self.stream.grab()
            if not grabbed: self.stop(); continue
            with self.lock: _, self.frame = self.stream.retrieve()
    def read(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None
    def stop(self): self.stopped = True; self.stream.release()

def send_command_to_pi(servo_id):
    try: requests.get(f"{URL_CONTROL}?id={servo_id}", timeout=0.05)
    except: pass 

# --- MAIN ---
print(f"🚀 CHẾ ĐỘ TRACKING: NHỚ LỖI TỪ XA")
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(str(MODEL_PATH))
kernel = np.ones((3, 3), np.uint8)

video_getter = VideoReceiver(URL_VIDEO).start()
time.sleep(1.0)

last_kick_1 = 0
last_kick_2 = 0

while True:
    frame = video_getter.read()
    if frame is None: time.sleep(0.01); continue

    height, width, _ = frame.shape
    MID_POINT = width // 2
    cv2.line(frame, (MID_POINT, 0), (MID_POINT, height), (255, 255, 0), 2)

    # 1. DÙNG TRACK THAY VÌ DETECT (Để có ID)
    # persist=True: Giữ ID liên tục qua các khung hình
    results = model.track(frame, conf=CONF_THRESHOLD, device=device, persist=True, verbose=False, tracker="bytetrack.yaml")
    current_time = time.time()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            label = model.names[int(cls)]
            
            if label not in ['orange', 'apple', 'fruit', 'sports ball']: continue

            x1, y1, x2, y2 = map(int, box)
            
            # --- KIỂM TRA BỘ NHỚ ---
            # Xem quả này trước đây đã từng bị phát hiện lỗi chưa
            is_already_bad = fruit_memory[track_id]['is_bad']

            # Cắt ảnh quả để soi
            fruit_img = frame[y1:y2, x1:x2]
            
            current_frame_bad = False
            max_blob_current = 0

            if fruit_img.size > 0:
                # 2. XỬ LÝ ẢNH TÌM MỰC
                hsv = cv2.cvtColor(fruit_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > max_blob_current: max_blob_current = area
                    
                    # Nếu thấy lỗi -> Vẽ viền vàng ngay trên vết mực
                    if area > MIN_BLOB_AREA:
                        current_frame_bad = True
                        cv2.drawContours(fruit_img, [cnt], -1, (0, 255, 255), 2)

            # --- LOGIC "CHỐT SỔ" ---
            # Nếu hiện tại thấy lỗi HOẶC quá khứ đã từng lỗi
            if current_frame_bad or is_already_bad:
                
                # Cập nhật vào bộ nhớ (để các frame sau vẫn nhớ là nó hỏng)
                fruit_memory[track_id]['is_bad'] = True
                if max_blob_current > fruit_memory[track_id]['max_error']:
                    fruit_memory[track_id]['max_error'] = max_blob_current
                
                status = f"LOI (ID:{track_id})"
                color = (0, 0, 255) # Đỏ
                
                # LOGIC GẠT (mỗi track_id chỉ gửi lệnh một lần)
                center_x = (x1 + x2) // 2
                
                # Vẫn giữ delay theo từng bên để bảo vệ cơ cấu servo khi nhiều quả lỗi đi sát nhau
                # (Logic này đơn giản, nếu muốn chuẩn hơn phải dùng Line 2 như file ai_worker)
                if track_id not in sent_ids:
                    if center_x < MID_POINT:
                        if current_time - last_kick_1 > DELAY_GAT:
                            threading.Thread(target=send_command_to_pi, args=('1',)).start()
                            sent_ids.add(track_id)
                            last_kick_1 = current_time
                            print(f"🚨 GẠT TRÁI - ID {track_id} HỎNG TỪ XA")
                    else:
                        if current_time - last_kick_2 > DELAY_GAT:
                            threading.Thread(target=send_command_to_pi, args=('2',)).start()
                            sent_ids.add(track_id)
                            last_kick_2 = current_time
                            print(f"🚨 GẠT PHẢI - ID {track_id} HỎNG TỪ XA")

            else:
                # Quả sạch sẽ từ đầu đến giờ
                status = f"TOT (ID:{track_id})"
                color = (0, 255, 0) # Xanh

            # Vẽ khung kết quả
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("TRACKING & MEMORY (XU LY TU XA)", frame)
    if cv2.waitKey(1) == ord('q'): break

video_getter.stop()
cv2.destroyAllWindows()

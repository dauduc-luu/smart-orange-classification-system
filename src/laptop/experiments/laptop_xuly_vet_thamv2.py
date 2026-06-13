import cv2
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
import time
from collections import defaultdict

# --- CẤU HÌNH ---
PI_IP = "10.29.211.93"  # <--- CHECK IP
URL_VIDEO = f"http://{PI_IP}:5000/video"
URL_CONTROL = f"http://{PI_IP}:5000/control"

MODEL_PATH = 'yolov8n.pt'
CONF_THRESHOLD = 0.5 

# --- CẤU HÌNH BẮT VẾT THÂM (GIỮ NGUYÊN THÔNG SỐ CŨ) ---

# BỎ biến MIN_FRUIT_SIZE_TO_CHECK để soi từ xa

# Độ nhạy kích thước lỗi (Vết thâm thường nhỏ hơn vết mực)
MIN_BLOB_AREA = 80  

# Dải màu cho VẾT THÂM (Nâu/Xám/Đen)
LOWER_DARK = np.array([0, 20, 0])      
UPPER_DARK = np.array([180, 255, 110]) 

# Tỉ lệ gọt viền (Mask tròn) - Để tránh bắt nhầm bóng râm ở viền quả
EDGE_CROP_RATIO = 0.15 

# Thời gian chờ (Vì phát hiện từ xa nên cần delay lâu hơn chút)
DELAY_GAT = 2.0 

# --- BỘ NHỚ TRẠNG THÁI ---
# Lưu lại lịch sử "bệnh án" của từng quả
fruit_memory = defaultdict(lambda: {'is_bad': False, 'max_error': 0})

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
print(f"🚀 CHẾ ĐỘ TRACKING: BẮT VẾT THÂM TỪ XA")
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
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

    # 1. TRACKING (Cấp ID cho quả)
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
            
            # Kiểm tra xem quả này đã từng bị đánh dấu hỏng chưa
            is_already_bad = fruit_memory[track_id]['is_bad']

            # Cắt ảnh quả
            fruit_img = frame[y1:y2, x1:x2]
            
            current_frame_bad = False
            max_blob_current = 0

            if fruit_img.size > 0:
                # --- THUẬT TOÁN "GỌT VỎ" (MASK TRÒN) ---
                h_img, w_img, _ = fruit_img.shape
                mask_circle = np.zeros((h_img, w_img), dtype=np.uint8)
                
                # Tạo tâm và bán kính (thu nhỏ lại theo EDGE_CROP_RATIO)
                center_circle = (w_img // 2, h_img // 2)
                radius = int(min(h_img, w_img) / 2 * (1 - EDGE_CROP_RATIO)) 
                cv2.circle(mask_circle, center_circle, radius, 255, -1)

                # Chỉ lấy phần lõi
                fruit_core = cv2.bitwise_and(fruit_img, fruit_img, mask=mask_circle)

                # --- SOI MÀU TRÊN PHẦN LÕI ---
                hsv = cv2.cvtColor(fruit_core, cv2.COLOR_BGR2HSV)
                mask_dark = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)
                mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > max_blob_current: max_blob_current = area
                    
                    if area > MIN_BLOB_AREA:
                        current_frame_bad = True
                        # Vẽ viền Xanh Dương (Cyan) cho vết thâm
                        cv2.drawContours(fruit_img, [cnt], -1, (255, 255, 0), 2)
                
                # Vẽ vòng tròn an toàn để bạn căn chỉnh
                cv2.circle(frame, (x1 + w_img//2, y1 + h_img//2), radius, (200, 200, 200), 1)

            # --- LOGIC GHI NHỚ & GẠT ---
            if current_frame_bad or is_already_bad:
                # Cập nhật bộ nhớ
                fruit_memory[track_id]['is_bad'] = True
                if max_blob_current > fruit_memory[track_id]['max_error']:
                    fruit_memory[track_id]['max_error'] = max_blob_current
                
                status = f"THAM (ID:{track_id})"
                color = (0, 0, 255) # Đỏ
                
                center_x = (x1 + x2) // 2
                
                # Logic gạt (Delay)
                if center_x < MID_POINT:
                    if current_time - last_kick_1 > DELAY_GAT:
                        threading.Thread(target=send_command_to_pi, args=('1',)).start()
                        last_kick_1 = current_time
                        print(f"🚨 TRÁI - ID {track_id} BỊ THÂM TỪ XA")
                else:
                    if current_time - last_kick_2 > DELAY_GAT:
                        threading.Thread(target=send_command_to_pi, args=('2',)).start()
                        last_kick_2 = current_time
                        print(f"🚨 PHẢI - ID {track_id} BỊ THÂM TỪ XA")
            
            else:
                status = f"OK (ID:{track_id})"
                color = (0, 255, 0) # Xanh

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("TRACKING VET THAM (ROT)", frame)
    if cv2.waitKey(1) == ord('q'): break

video_getter.stop()
cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
import time

# --- CẤU HÌNH ---
PI_IP = "10.29.211.93"  # <--- CHECK IP
URL_VIDEO = f"http://{PI_IP}:5000/video"
URL_CONTROL = f"http://{PI_IP}:5000/control"

MODEL_PATH = 'yolov8n.pt'
CONF_THRESHOLD = 0.6 

# --- CẤU HÌNH BẮT VẾT THÂM/HỎNG ---

# 1. Kích thước quả để bắt đầu soi (như cũ)
MIN_FRUIT_SIZE_TO_CHECK = 15000 

# 2. Độ nhạy kích thước lỗi (QUAN TRỌNG)
# Giảm xuống thấp để bắt vết thâm nhỏ.
# Trước để 300, giờ giảm xuống 50 hoặc 80 để bắt vết dập bé.
MIN_BLOB_AREA = 80  

# 3. Dải màu cho VẾT THÂM (Nâu/Xám/Đen)
# LOWER: Vẫn giữ đen
# UPPER: Tăng độ sáng (V) từ 60 lên 110 (để bắt được màu nâu/thâm)
LOWER_DARK = np.array([0, 20, 0])     # Saturation > 20 để bỏ qua màu trắng lóa
UPPER_DARK = np.array([180, 255, 110]) # Brightness < 110 (Màu tối/thâm)

# 4. TỈ LỆ GỌT VIỀN (PADDING) - CỰC QUAN TRỌNG
# Số 0.15 nghĩa là bỏ qua 15% diện tích viền xung quanh quả (vì viền hay bị tối do bóng)
# Tăng lên nếu máy hay bắt nhầm viền quả.
EDGE_CROP_RATIO = 0.15 

DELAY_GAT = 2.0

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
print(f"🚀 CHẾ ĐỘ 5: BẮT VẾT THÂM/DẬP (ROT DETECTION)")
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)

# Kernel xử lý ảnh
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

    # 1. AI TÌM QUẢ
    results = model(frame, conf=CONF_THRESHOLD, device=device, verbose=False)
    current_time = time.time()

    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        
        if label in ['orange', 'apple', 'fruit', 'sports ball']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            
            # Chỉ soi khi quả đến gần
            if box_area < MIN_FRUIT_SIZE_TO_CHECK:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                continue 

            # Cắt ảnh quả
            fruit_img = frame[y1:y2, x1:x2]
            if fruit_img.size == 0: continue
            
            # >>> KỸ THUẬT MỚI: TẠO MẶT NẠ TRÒN ĐỂ GỌT VIỀN <<<
            # Mục đích: Che đi phần viền tối của quả để không bắt nhầm
            h_img, w_img, _ = fruit_img.shape
            mask_circle = np.zeros((h_img, w_img), dtype=np.uint8)
            
            # Vẽ hình tròn màu trắng ở giữa (nhỏ hơn quả một chút)
            center_circle = (w_img // 2, h_img // 2)
            radius = int(min(h_img, w_img) / 2 * (1 - EDGE_CROP_RATIO)) # Bán kính nhỏ lại
            cv2.circle(mask_circle, center_circle, radius, 255, -1)

            # Áp dụng mặt nạ tròn lên ảnh gốc (Chỉ giữ lại phần lõi quả)
            fruit_core = cv2.bitwise_and(fruit_img, fruit_img, mask=mask_circle)
            
            # 2. SOI VẾT THÂM TRÊN PHẦN LÕI
            hsv = cv2.cvtColor(fruit_core, cv2.COLOR_BGR2HSV)
            mask_dark = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)
            
            # Xử lý nhiễu
            mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel)
            
            # Tìm các cục vết thâm
            contours, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            is_bad = False
            max_blob_area = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_blob_area: max_blob_area = area
                
                # Bắt vết dập nhỏ hơn (vì giờ mình đã giảm ngưỡng)
                if area > MIN_BLOB_AREA:
                    is_bad = True
                    # Vẽ viền Xanh Dương (Cyan) quanh vết thâm để phân biệt
                    cv2.drawContours(fruit_img, [cnt], -1, (255, 255, 0), 2)

            center_x = (x1 + x2) // 2

            # 3. RA QUYẾT ĐỊNH
            if is_bad:
                status = f"THAM/HONG ({int(max_blob_area)})"
                color = (0, 0, 255) # Đỏ
                
                if center_x < MID_POINT:
                    if current_time - last_kick_1 > DELAY_GAT:
                        threading.Thread(target=send_command_to_pi, args=('1',)).start()
                        last_kick_1 = current_time
                        print(f"🚨 TRÁI: Phát hiện vết thâm: {max_blob_area}")
                else:
                    if current_time - last_kick_2 > DELAY_GAT:
                        threading.Thread(target=send_command_to_pi, args=('2',)).start()
                        last_kick_2 = current_time
                        print(f"🚨 PHẢI: Phát hiện vết thâm: {max_blob_area}")
            else:
                status = f"OK"
                color = (0, 255, 0) # Xanh

            # Vẽ vòng tròn an toàn (để bạn biết vùng nào đang được soi)
            cv2.circle(frame, (x1 + w_img//2, y1 + h_img//2), radius, (200, 200, 200), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("LAPTOP - ROT DETECTION", frame)
    if cv2.waitKey(1) == ord('q'): break

video_getter.stop()
cv2.destroyAllWindows()
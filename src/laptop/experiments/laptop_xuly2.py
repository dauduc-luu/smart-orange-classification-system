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

# --- CẤU HÌNH THÔNG MINH MỚI ---

# 1. KÍCH THƯỚC QUẢ ĐỂ BẮT ĐẦU SOI (Pixel)
# Quả phải to hơn số này (tức là đã đi đến gần) mới bắt đầu soi
# Để tránh soi nhầm mấy quả ở tít đằng xa
MIN_FRUIT_SIZE_TO_CHECK = 15000 

# 2. TỈ LỆ LỖI (% Diện tích)
# Vết mực phải chiếm trên 1.5% diện tích quả thì mới gạt
# Cái này giúp loại bỏ chấm li ti dù quả đang ở rất gần
MIN_ERROR_PERCENT = 1.5 

# 3. MÀU MỰC
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 50]) # Giảm độ sáng xuống 70 để đỡ bắt nhầm bóng râm

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
print(f"🚀 Đang chạy chế độ: CHỐNG SAI KHOẢNG CÁCH")
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
kernel = np.ones((5, 5), np.uint8) 

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
            
            # Tính diện tích khung hình quả (Box Area)
            box_area = (x2 - x1) * (y2 - y1)
            
            # >>> LUẬT 1: CHỈ SOI KHI QUẢ ĐÃ ĐẾN GẦN <<<
            if box_area < MIN_FRUIT_SIZE_TO_CHECK:
                # Nếu quả còn bé (ở xa), vẽ khung xám và bỏ qua
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue 

            # Cắt ảnh
            fruit_img = frame[y1:y2, x1:x2]
            if fruit_img.size == 0: continue
            
            # Xử lý màu & Lọc nhiễu
            hsv = cv2.cvtColor(fruit_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Đếm điểm lỗi
            ink_pixels = cv2.countNonZero(mask)
            
            # >>> LUẬT 2: TÍNH PHẦN TRĂM (%) <<<
            # Lỗi chiếm bao nhiêu phần trăm so với diện tích quả?
            error_ratio = (ink_pixels / box_area) * 100
            
            center_x = (x1 + x2) // 2

            # KIỂM TRA NGƯỠNG %
            if error_ratio > MIN_ERROR_PERCENT:
                status = f"LOI ({error_ratio:.1f}%)"
                color = (0, 0, 255) # Đỏ
                
                if center_x < MID_POINT:
                    if current_time - last_kick_1 > DELAY_GAT:
                        threading.Thread(target=send_command_to_pi, args=('1',)).start()
                        last_kick_1 = current_time
                        print(f" GẠT TRÁI - Lỗi: {error_ratio:.1f}%")
                else:
                    if current_time - last_kick_2 > DELAY_GAT:
                        threading.Thread(target=send_command_to_pi, args=('2',)).start()
                        last_kick_2 = current_time
                        print(f" GẠT PHẢI - Lỗi: {error_ratio:.1f}%")
            else:
                status = f"OK ({error_ratio:.1f}%)"
                color = (0, 255, 0) # Xanh

            # Vẽ khung
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Debug: Hiện mask để xem nó bắt cái gì
            # cv2.imshow("Mask", mask)

    cv2.imshow("He thong chong loai bo sai", frame)
    if cv2.waitKey(1) == ord('q'): break

video_getter.stop()
cv2.destroyAllWindows()	
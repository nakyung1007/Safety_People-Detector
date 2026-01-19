# 표준 라이브러리
import sys
import os
import time
import csv
import math
import struct
import socket
import threading
import traceback
from datetime import datetime
from queue import Queue, Empty
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# 서드파티 라이브러리
import cv2
import serial
import numpy as np
import datetime as dt
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator, AutoLocator
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
from sklearn.cluster import DBSCAN
# PyQt5

from PyQt5.QtCore import (
    Qt, QTimer, QDateTime, QRectF, QPoint, pyqtSignal, QThread
)
from PyQt5.QtGui import (
    QPixmap, QFont, QImage, QPainter, QPolygon, QColor,
    QKeySequence, QRegion, QPainterPath, QBrush, QTransform, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QFrame, QPushButton, QStackedWidget, QComboBox, QMenu, QAction,
    QShortcut, QMessageBox, QLineEdit
)

from safety_logger import *
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("[SYSTEM] YOLO 모듈 로드 성공")
except ImportError:
    YOLO_AVAILABLE = False
    print("[SYSTEM] YOLO 모듈 없음 - 일반 카메라 모드로 동작")


HOST, PORT = "192.168.0.99", 2111
STX, ETX = b"\x02", b"\x03"

#SICK ASCII 프로토콜의 프레임 시작(STX), 끝(ETX)
# 모든 명령은 STX + 명령 + ETX 형태로 전송해야 함

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # TCP소켓 생성 
# 수정 전
# sock.connect((HOST, PORT))

# 수정 후 (예외 처리 추가)
try:
    sock.settimeout(3.0) # 3초 동안 기다려봄
    sock.connect((HOST, PORT))
    print("LiDAR 연결 성공!")
except socket.timeout:
    print("에러: 라이더가 대답이 없습니다. IP 설정이나 케이블을 확인하세요.")
except Exception as e:
    print(f"에러 발생: {e}")
    
sock.sendall(STX + b"sEN LMDscandata 1" + ETX)  # 스캔데이터 송신
sock.settimeout(1.0)    #1초이상 응답 없다면 프로그램 멈춤을 방지함

#OUT_DIR = r"C:\Users\admin\Desktop\카메라"    # 카메라 저장경로
60
OUT_DIR = r"C:\Users\whskr\Desktop\merzes\Motor_BlackBox"

def resource_path(rel_path):
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS      # exe 내부 리소스
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, rel_path)

# 헥사 변환 함수
def hex_to_int32_signed(h: str) -> int:
    v = int(h, 16)
    return v - (1 << 32) if v & (1 << 31) else v

def hex_to_float32(h: str) -> float:
    return struct.unpack(">f", bytes.fromhex(h))[0]

# LMD 파싱
def parse_lmd(frame):
    if "DIST1" not in frame:
        return None, None

    toks = frame.split()
    try:
        i = toks.index("DIST1")
        scale = hex_to_float32(toks[i + 1])
        offset = hex_to_float32(toks[i + 2])
        start_deg = hex_to_int32_signed(toks[i + 3]) / 10000.0
        step_deg  = int(toks[i + 4], 16) / 10000.0
        n_pts     = int(toks[i + 5], 16)

        vals = toks[i + 6 : i + 6 + n_pts]
        dist = np.array([int(v, 16) * scale / 1000.0 + offset for v in vals])
        angles = np.deg2rad(start_deg + np.arange(n_pts) * step_deg)
        return angles, dist
    except:
        return None, None

# 사람 판별
def is_person(pts):
    n = len(pts)
    if not (20 <= n <= 70):
        return False

    width  = np.ptp(pts[:,0])
    height = np.ptp(pts[:,1])
    if width > 1.2 or height > 1.2:
        return False

    ratio = height / (width + 1e-6)
    if not (0.3 <= ratio <= 4.0):
        return False

    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    if np.std(d) < 0.003:
        return False

    return True

class CameraThread(QThread):
    frame_signal = pyqtSignal(object)
    fps_signal = pyqtSignal(float)

    # 수정 후:
    def __init__(self, ui_ref, rtsp_url, out_dir, fourcc="mp4v", target_fps=30,
                yolo_model_path=None, enable_yolo=False):
        super().__init__()
        self.ui = ui_ref
        self.rtsp_url = rtsp_url
        self.base_out_dir = out_dir
        self.fourcc = fourcc
        self.target_fps = target_fps

        self.running = True
        self.recording = False
        self.writer = None
        self.frame_size = None

        self.segment_sec = 60
        self.segment_start_time = None

        # ★★★ YOLO 추가 부분 ★★★
        self.enable_yolo = enable_yolo and YOLO_AVAILABLE
        self.yolo_model = None
        
        if self.enable_yolo and yolo_model_path:
            try:
                from ultralytics import YOLO
                import torch # 상단에 import 필요
                if os.path.exists(yolo_model_path):
                    # GPU 사용 가능 여부 확인 후 장치 할당
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.yolo_model = YOLO(yolo_model_path).to(self.device) # GPU로 모델 이동
                    print(f"[CAM] YOLO 모델 로드 성공 ({self.device} 모드): {yolo_model_path}")
                else:
                    self.enable_yolo = False
            except Exception as e:
                print(f"[CAM] YOLO 로드 실패: {e}")
                self.enable_yolo = False

    # 날짜별 폴더 생성
    def get_today_folder(self):
        today = dt.datetime.now().strftime("%Y%m%d")
        folder = os.path.join(self.base_out_dir, today)

        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

    # VideoWriter 생성 (1분 단위 파일)
    def create_writer(self):
        if self.frame_size is None:
            print("[ERROR] frame_size 없음")
            return None

        out_dir = self.get_today_folder()

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cam_{ts}.mp4"
        save_path = os.path.join(out_dir, filename)

        w, h = self.frame_size

        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.target_fps,
            (w, h),
        )
        if not writer.isOpened():
            print("[ERROR] VideoWriter 열기 실패")
            return None

        print(f"[REC] 저장 시작: {save_path}")
        return writer

    
    # data save버튼 
    def start_recording(self):
        if self.recording:
            return  # 이미 녹화 중

        self.writer = self.create_writer()
        if self.writer is None:
            return

        self.recording = True
        self.segment_start_time = time.time()
        print("[REC] 녹화 ON")

    
    # data save stop버튼
    def stop_recording(self):
        self.recording = False
        print("[REC] 녹화 OFF")

        # 더미(마지막) 프레임 추가 > pts <= last 방지
        try:
            if self.writer:
                black = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                self.writer.write(black)
        except:
            pass

        if self.writer:
            try:
                self.writer.release()
            except:
                pass
            self.writer = None


    def run(self):

        # Camera open 보호
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print("[ERROR] Camera open exception:", e)
            return

        if not self.cap.isOpened():
            print("[ERROR] 카메라 열기 실패")
            return

        start_time = time.time()
        frame_count = 0

        # print("[CAM] Thread started")

        while self.running:

            try:
                
                # 프레임 읽기 (OpenCV 예외 방어)
                
                try:
                    ok, frame = self.cap.read()
                except cv2.error as e:
                    print("[CAM] OpenCV read exception:", e)
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print("[CAM] read unknown exception:", e)
                    time.sleep(0.1)
                    continue

                if not ok or frame is None:
                    self.ui.state_icon.camera_connected = False
                    print("[CAM] frame read failed")
                    time.sleep(0.5)
                    continue

                # 정상 프레임
                self.ui.last_camera_frame_time = time.time()
                self.ui.state_icon.camera_connected = True

                
                # 최초 frame_size 설정
                
                if self.frame_size is None:
                    h, w = frame.shape[:2]
                    self.frame_size = (w, h)
                
                  # ★★★ YOLO 추론 추가 ★★★
                if self.enable_yolo and self.yolo_model:
                    try:
                        results = self.yolo_model.predict(
                            frame,
                            conf=0.5,
                            iou=0.45,
                            verbose=False,
                            device= self.device
                        )
                        
                        # --- [추가] 로그용 데이터를 담을 임시 리스트 ---
                        temp_log_data = []

                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                                # 신뢰도 및 클래스
                                conf = float(box.conf[0])
                                cls_name = self.yolo_model.names[int(box.cls[0])]
                                
                                # 바운딩 박스 그리기 (빨간색)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                
                                # 라벨 텍스트
                                label = f"{cls_name} {conf:.2f}"
                                
                                # 라벨 배경
                                (tw, th), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                                )
                                cv2.rectangle(
                                    frame,
                                    (x1, y1 - th - 10),
                                    (x1 + tw, y1),
                                    (0, 0, 255),
                                    -1
                                )
                                
                                # 라벨 텍스트
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 255, 255),
                                    2
                                )
                                temp_log_data.append({
                                    "class":cls_name,
                                    "conf": round(conf,2),
                                    "coords" : {"x1": x1, "y1" :y1, "x2": x2, "y2": y2}
                                })
                        self.ui.cuurent_detected_people = temp_log_data
                    except Exception as e:
                        print(f"[CAM] YOLO 추론 에러: {e}")
                # ★★★ YOLO 추론 끝 ★★★

                
                # 녹화 중이면 writer write
                
                if self.recording:

                    # writer 유효성 검사
                    if self.writer is None or not self.writer.isOpened():
                        print("[CameraThread] WARNING: writer invalid > recreate")
                        self.writer = self.create_writer()
                        if self.writer is None or not self.writer.isOpened():
                            print("[CameraThread] writer still invalid > skip frame")
                            continue

                    # 안전 write
                    try:
                        self.writer.write(frame)
                    except Exception as e:
                        print("[CameraThread] ERROR write:", e)
                        try:
                            self.writer.release()
                        except:
                            pass
                        self.writer = self.create_writer()
                        continue

                    #분 rotate
                    if time.time() - self.segment_start_time >= self.segment_sec:
                        try:
                            self.writer.release()
                        except:
                            pass
                        self.writer = self.create_writer()
                        self.segment_start_time = time.time()

                
                # FPS 계산
                
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    self.fps_signal.emit(fps)
                    start_time = time.time()
                    frame_count = 0

                
                # QImage 변환 및 emit
                
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_signal.emit(qimg)
                except Exception as e:
                    print("[CameraThread] ERROR convert/send frame:", e)
                    continue

            except Exception as e:
                # 어떤 예상치 못한 예외도 전체 스레드를 죽이지 않는다.
                print("[CameraThread] UNCAUGHT ERROR:", e)
                time.sleep(0.2)
                continue

        
        # 스레드 종료 정리
        
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            self.cap.release()

        if self.writer:
            try:
                self.writer.release()
            except:
                pass

        print("[CAM] Thread stopped")


    def stop(self):
        self.running = False

        # 카메라 해제
        try:
            if hasattr(self, "cap") and self.cap.isOpened():
                self.cap.release()
                print("[CAM] cap.release()")
        except:
            pass

        # 녹화 종료
        try:
            if self.writer:
                self.writer.release()
                print("[CAM] writer.release()")
        except:
            pass

        self.wait()


# PyQt 위젯용 LiDAR CANVAS
class LidarCanvas(FigureCanvas):
    coord_signal = pyqtSignal(float, float, float, float)
    def __init__(self, ui_ref, parent=None):
        
        
        
        self.fig = Figure(figsize=(8.6, 5.4), dpi=100)
        super().__init__(self.fig)
        self.ui = ui_ref       
        self.setParent(ui_ref)

        self.lidar_save_enabled = False
        self.csv_file = None
        self.csv_writer = None

        
        self.csv_segment_sec = 60   #분 단위 분할
        self.csv_segment_start = None


        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-25, 25)
        self.ax.set_ylim(-25, 25)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.xaxis.set_major_locator(MultipleLocator(5))
        self.ax.yaxis.set_major_locator(MultipleLocator(5))
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_title("TiM571", pad=20)
        self.ax.xaxis.set_ticks_position("top")
        self.ax.xaxis.set_label_position("top")

        self.setStyleSheet("background: transparent;")
        self.fig.patch.set_facecolor("none")
        # self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("none")
        
        # FOV
        fov = Wedge((0,0), 25, -45, 225, facecolor="lightgray", alpha=0.3)
        self.ax.add_patch(fov)

        # LiDAR marker
        self.ax.plot(0,0,"ks", markersize=8)

        self.scat = self.ax.scatter([], [], s=8)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.mouse_inside = False

        # 마우스 좌표 저장
        self.mouse_pos = [0, 0]

        # 회전 보정
        self.ANGLE_OFFSET = np.deg2rad(270.0)

        # 마우스 이벤트
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # 업데이트 시작
        self.ani = FuncAnimation(self.fig, self.update_lidar, interval=80, blit=False, cache_frame_data=False)

    def on_mouse_move(self, event):

        if event.inaxes is None:
            return

        # 좌표 없음 > 무시
        if event.xdata is None or event.ydata is None:
            return

        # 실제 좌표 계산
        x = event.xdata
        y = event.ydata
        d = (x**2 + y**2)**0.5
        beta = np.degrees(np.arctan2(y, x))

        # MyUI로 전달
        self.coord_signal.emit(x, y, d, beta)

    # 마우스 휠로 확대/축소
    def on_scroll(self, event):
        if event.xdata is None:
            return

        base = 1.2
        sx = (1 / base) if event.button == 'up' else base

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        w = (xlim[1] - xlim[0]) * sx
        h = (ylim[1] - ylim[0]) * sx

        rx = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
        ry = (event.ydata - ylim[0]) / (ylim[1] - ylim[0])

        self.ax.set_xlim([event.xdata - w * rx, event.xdata + w * (1 - rx)])
        self.ax.set_ylim([event.ydata - h * ry, event.ydata + h * (1 - ry)])

        self.ax.xaxis.set_major_locator(AutoLocator())
        self.ax.yaxis.set_major_locator(AutoLocator())

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.xdata is None:
            return
        self._pan = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, event):
        if not hasattr(self, "_pan") or event.inaxes != self.ax:
            return
        x0, y0, (x1, x2), (y1, y2) = self._pan
        dx = event.xdata - x0
        dy = event.ydata - y0
        self.ax.set_xlim(x1 - dx, x2 - dx)
        self.ax.set_ylim(y1 - dy, y2 - dy)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if hasattr(self, "_pan"):
            del self._pan
    
    # 업데이트
    def update_lidar(self, _):

        # LiDAR 데이터 수신 
        buf = b""
        try:
            buf += sock.recv(65536)
        except:
            # 수신 실패 > LiDAR 끊김 표시
            self.ui.state_icon.lidar_connected = False

            return self.scat,

        if STX not in buf or ETX not in buf:
            # 패킷 불량
            self.ui.state_icon.lidar_connected = False

            return self.scat,

        a, b = buf.find(STX), buf.find(ETX)
        msg = buf[a+1:b].decode(errors="ignore")

        if "LMDscandata" not in msg:
            self.ui.state_icon.lidar_connected = False

            return self.scat,

        th, r = parse_lmd(msg)
        if th is None:
            self.ui.state_icon.lidar_connected = False

            return self.scat,

        # 상태만 업데이트
        self.ui.state_icon.lidar_connected = True
        self.ui.last_lidar_frame_time = time.time()


        # 극>직교 변환
        th = (th + self.ANGLE_OFFSET) % (2*np.pi)
        x = -r * np.sin(th)
        y =  r * np.cos(th)
        pts = np.column_stack((x, y))

        # DBSCAN
        db = DBSCAN(eps=0.12, min_samples=5).fit(pts)
        labels = db.labels_

        # 색상 및 군집별 분류 
        colors = []
        for _, lab in enumerate(labels):
            if lab == -1:
                colors.append("gray")
            else:
                cluster = pts[labels == lab]
                colors.append("red" if is_person(cluster) else "blue")

        # 사람 centroid 계산 
        person_centroids = []
        for lab in set(labels):
            if lab == -1:
                continue
            cluster = pts[labels == lab]
            if is_person(cluster):
                cx, cy = cluster.mean(axis=0)
                person_centroids.append((cx, cy))

        # x 기준 정렬 (p1,p2 번호 고정)
        person_centroids = sorted(person_centroids, key=lambda p: p[0])

        # 산점도 업데이트 
        self.scat.set_offsets(pts)
        self.scat.set_color(colors)

        # CSV 기록 (data save 버튼 ON일 때만) 
        if self.lidar_save_enabled:
            
            if time.time() - self.csv_segment_start >= self.csv_segment_sec:
            # 오래된 파일 닫기
                if self.csv_file:
                    self.csv_file.close()

                # 새 파일 생성
                ts = time.strftime("%Y%m%d_%H%M%S")
                csv_filename = f"lidar_{ts}.csv"
                self.csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(["timestamp", "P1_x", "P1_y"])

                self.csv_segment_start = time.time()
                print(f"[LIDAR CSV] 새 파일 생성: {csv_filename}")
            
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            num_p = len(person_centroids)

            if num_p == 0:
                self.csv_writer.writerow([ts, "NONE"])
            else:
                coord_flat = []
                for (px, py) in person_centroids:
                    coord_flat.append(f"{px:.3f}")
                    coord_flat.append(f"{py:.3f}")

                # 사람 간 거리 (cm)
                dist_list = []
                for i in range(num_p):
                    for j in range(i+1, num_p):
                        dx = person_centroids[i][0] - person_centroids[j][0]
                        dy = person_centroids[i][1] - person_centroids[j][1]
                        dist_cm = math.sqrt(dx*dx + dy*dy) * 100
                        dist_list.append(f"(p{i+1}-p{j+1}){dist_cm:.2f}")

                self.csv_writer.writerow([ts] + coord_flat + dist_list)


        # 2. [위치 이동] 자동 추적 로직 (return 위로 올림)
        if self.ui.auto_tracking_enabled and person_centroids:
            target_p = person_centroids[0]  # 가장 먼저 발견된 사람 추적
            tx, ty = target_p[0], target_p[1]
            
            b_angle, t_angle = self.ui.tracker.calculate_angles(tx, ty)
            
            if self.ui.tracker.should_move(b_angle, t_angle):
                self.ui.sendCmd(f"s1:{b_angle}")
                time.sleep(0.01) 
                self.ui.sendCmd(f"s2:{t_angle}")
                
        # Info UI 업데이트 
        if self.mouse_inside:
            mx, my = self.mouse_pos
            d = math.sqrt(mx*mx + my*my)
            beta = math.degrees(math.atan2(my, mx))

            self.info_text.set_text(
                f"X: {mx:.3f} m\n"
                f"Y: {my:.3f} m\n"
                f"d: {d:.3f} m\n"
                f"β: {beta:.3f}°"
            )
        return self.scat,

class State_Icon:
    def __init__(self, ui_ref):
        self.ui = ui_ref  # UI를 갱신하기 위한 참조

        # 초기 상태 (전부 False > RED 상태)
        self.comport_connected = False
        self.camera_connected = False
        self.lidar_connected = False
        self.system_started = False
    
    # 상태 변경 함수들
    
    def set_comport(self, is_connected):
        self.comport_connected = is_connected
        self.update_state()

    def set_camera(self, is_connected):
        self.camera_connected = is_connected
        self.update_state()

    def set_lidar(self, is_connected):
        self.lidar_connected = is_connected
        self.update_state()

    def reset_all(self):
        self.comport_connected = False
        self.camera_connected = False
        self.lidar_connected = False
        self.update_state()

    
    # 상태 판정
    # def update_state(self):
    #     if self.comport_connected and self.camera_connected and self.lidar_connected:
    #         self.show_green()
    #     else:
    #         self.show_red()
    def update_state(self):
        # START 안 됐으면 무조건 RED
        if not self.system_started:
            self.show_red()
            return
        
        if (self.system_started and self.comport_connected and self.camera_connected and self.lidar_connected): 
            self.show_green()
        else:
            self.show_red()


    
    # 아이콘 표시
    def show_red(self):
        image_path = resource_path("img/그룹 78.png")
        pixmap = QPixmap(image_path)

        pixmap = pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.state_label_circle.setPixmap(pixmap)


    def show_green(self):
        image_path = resource_path("img/그룹 79.png")
        pixmap = QPixmap(image_path)

        pixmap = pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.state_label_circle.setPixmap(pixmap)


class RoundedMenu(QMenu):
    def showEvent(self, event):
        super().showEvent(event)
        path = QPainterPath()
        rect = QRectF(0, 0, self.width(), self.height())
        radius = 15
        path.addRoundedRect(rect, radius, radius)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

class ToggleButton(QPushButton):
    clicked_release = pyqtSignal()  # 우리가 직접 만든 클릭 시그널

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # 버튼 영역 안에서 손을 뗀 경우에만 시그널 발생
        if self.rect().contains(event.pos()):
            self.clicked_release.emit()

class MyUI(QWidget):
    def __init__(self):
        super().__init__()
        self.state_icon = State_Icon(self)
        self.initUI()
        self.menu_open = False
        self.selected_port = None
        self.serial_conn = None
        self.ser = None               # 시리얼 객체
        self.last_camera_frame_time = time.time()
        self.last_lidar_frame_time = time.time()
        self.device_check_timer = QTimer()
        self.device_check_timer.timeout.connect(self.check_device_status)
        self.device_check_timer.start(2000)   #초마다 검사
        # [여기 추가] 로그 관련 초기화
        self.safety_logger = SafetyLogger(OUT_DIR)
        self.current_detected_people = []  # 탐지된 사람 정보 보관함
        
        # [여기 추가] 5초 로그 타이머
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.save_5sec_safety_log)

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.update_timestamp)
        self.timer2.start(1000)
        self.update_timestamp()

    def save_5sec_safety_log(self):
        """5초마다 실행될 실제 저장 로직"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "safety_status": {
                "min_distance": getattr(self, 'current_min_distance', None),
                "risk_level": "SAFE" 
            },
            "detected_people": self.current_detected_people, # 보관함에서 꺼내옴
            "detected_equipment": [] 
        }
        self.safety_logger.write_json_log(log_entry)
        print(f"[SYSTEM] JSON 로그 저장 완료")

    def toggle_comport_menu(self):
        try:
            # 이미 메뉴가 떠 있으면 닫기
            if self.comport_menu.isVisible():
                self.comport_menu.hide()
                return

            # 메뉴 열기
            self.populate_comport_menu()
            pos = self.btn_comport_set.mapToGlobal(
                self.btn_comport_set.rect().bottomLeft()
            )
            self.comport_menu.popup(pos)

        except Exception as e:
            print("toggle error:", e)
            

    def update_camera_frame(self, qimg):
        pix = QPixmap.fromImage(qimg)

        rounded = QPixmap(pix.size())
        rounded.fill(Qt.transparent)

        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)

        radius = 15
        path = QPainterPath()

        w = pix.width()
        h = pix.height()

        # 아래쪽만 둥글게 모서리 적용
        path.moveTo(0, 0)
        path.lineTo(0, h - radius)
        path.quadTo(0, h, radius, h)
        path.lineTo(w - radius, h)
        path.quadTo(w, h, w, h - radius)
        path.lineTo(w, 0)
        path.lineTo(0, 0)

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pix)
        painter.end()

        self.camera_label.setPixmap(rounded)
        
        self.state_icon.camera_connected = True
        self.last_camera_frame_time = time.time()      # 최근 수신 시각 업데이트
        

    def update_fps_label(self):
        pass

    def update_coords(self, x, y, d, beta):
        self.coord_x_label.setText(f"{x:.3f}m")
        self.coord_y_label.setText(f"{y:.3f}m")
        self.coord_d_label.setText(f"{d:.3f}m")
        self.coord_b_label.setText(f"{beta:.2f}°")


    def _menu_closed_block(self):
            self.menu_open = False
            self.btn_comport_set.blockSignals(True)
            QTimer.singleShot(200, lambda: self.btn_comport_set.blockSignals(False))

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()        # 전체화면 > 일반창
        else:
            self.showFullScreen()    # 일반창 > 전체화면

    def initUI(self):
        self.setWindowTitle("Motor_BlackBox")
        self.setGeometry(0, 0, 1920, 1080)  # 메인 윈도우 크기
        self.setStyleSheet("background-color: #FCFBFB;")

        self.logo_image_label = QLabel(self)
        self.logo_image_label.setGeometry(80, 20, 260, 50)

        # F11 전체화면 토글
        self.shortcut_fullscreen = QShortcut(QKeySequence("F11"), self)
        self.shortcut_fullscreen.activated.connect(self.toggle_fullscreen)

        logo_image_path = resource_path("img/로고 02.png")
        pixmap = QPixmap(logo_image_path)
        self.logo_image_label.setPixmap(
            pixmap.scaled(260, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # State 이미지 출력
        self.state_label = QLabel(self)
        self.state_label.setGeometry(780, 37, 60, 26)
        self.state_label.setStyleSheet("background: transparent;")

        image_path = resource_path("img/State.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(60, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.state_label.setPixmap(pixmap)


        # State 상태 이미지 출력
        self.state_label_circle = QLabel(self)
        self.state_label_circle.setGeometry(925, 40, 20, 20)
        self.state_label_circle.setStyleSheet("background: transparent;")

        self.state_icon = State_Icon(self)
        self.state_icon.update_state() 

        self.log_timestamp = QLabel(self)
        self.log_timestamp.setGeometry(1602, 35, 248, 30)
        self.log_timestamp.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.log_timestamp.setStyleSheet("font-family: 'NanumSquareOTF'; font-size: 27px; color: #000000;")

        # CAM 이미지 출력
        self.cam_label = QLabel(self)
        self.cam_label.setGeometry(70,90,880,40)
        self.cam_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/그룹 77.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(880,40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam_label.setPixmap(pixmap)


        # 사각형 32 라벨
        self.rect32_label = QLabel(self)
        self.rect32_label.setGeometry(70, 130, 880, 420)
        self.rect32_label.setStyleSheet("""
            background: transparent;
            border:  solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        image_path = resource_path("img/사각형 32.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(880, 420, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect32_label.setPixmap(pixmap)


        # 카메라 영상 출력 영역
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(70, 130, 880, 420)
        self.camera_label.setScaledContents(True)
        self.camera_label.setStyleSheet("""
            background: transparent;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        self.camera_label.raise_()

        # MAP 이미지 출력
        self.map_label = QLabel(self)
        self.map_label.setGeometry(990,90,860,40)
        self.map_label.setStyleSheet("background: transparent;")

        image_path = resource_path("img/그룹 76.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(860,40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.map_label.setPixmap(pixmap)

        # 사각형 48 라벨
        self.rect48_label = QLabel(self)
        self.rect48_label.setGeometry(990, 130, 860, 540)
        self.rect48_label.setStyleSheet("""
            background: transparent;
            border: solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        image_path = resource_path("img/사각형 48.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(860, 540, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect48_label.setPixmap(pixmap)


        self.lidar_canvas = LidarCanvas(self)
        self.lidar_canvas.setGeometry(990, 130, 860, 540)
        self.lidar_canvas.raise_()


        self.log_set_servo = QLabel("Set Servo Motor", self)
        self.log_set_servo.setGeometry(416,588,188,26)
        self.log_set_servo.setStyleSheet("font-family: 'NanumSquareOTF'; font-size: 24px; font-weight: 800; color: #1C426D; text-align: center;")

        self.log_set_servo_comport = QLabel("Comport", self)
        self.log_set_servo_comport.setGeometry(312,651,96,26)
        self.log_set_servo_comport.setStyleSheet("font-family: 'NanumSquareOTF'; font-size: 24px; font-weight: normal; color: #000000;")

        #COMPORT SET 버튼
        self.btn_comport_set = ToggleButton("Comport Set",self)
        self.btn_comport_set.setGeometry(468, 634, 240, 60)
        #순위 해결: auto-repeat 완전 차단
        self.btn_comport_set.setAutoRepeat(False)
        comport_bg = resource_path("img/rmfnq 71.png").replace("\\", "/")
        self.btn_comport_set.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background: transparent url('{comport_bg}') 0% 0% no-repeat;
                font-family: 'NanumSquareOTF';
                font-size: 24px;
                color: #000000;
                padding-right: 20px;  
            }}
            QPushButton:pressed {{
                background: #F2F2F2;
                border: 1px solid #CCCCCC;
                border-radius: 15px;
            }}
        """)
        self.btn_comport_set.setCursor(Qt.PointingHandCursor)
        self.btn_comport_set.clicked_release.connect(self.toggle_comport_menu)

        # 포트 목록 메뉴 생성 (btn_comport_set 정의 후 생성)
        self.comport_menu = RoundedMenu(self)
        self.comport_menu.aboutToHide.connect(self._menu_closed_block)
        self.comport_menu.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.comport_menu.setAttribute(Qt.WA_TranslucentBackground)
        comport_bg2 = resource_path("img/사각형 131.png").replace("\\", "/")
        self.comport_menu.setStyleSheet(f"""
            QMenu {{
                background: transparent url('{comport_bg2}') 0% 0% no-repeat;
                border: 1px solid #CCCCCC;
                border-radius: 15px;
            }}
            QMenu::item {{
                padding: 10px 20px;
                background-color: transparent;
                font-family: 'NanumSquareOTF';
                font-size: 24px;
                color: #000000;
            }}
            QMenu::item:selected {{
                background-color: #E6E6E6;
                border-radius: 15px;
            }}
        """)

        self.populate_comport_menu()
        self.comport_menu.setFixedWidth(self.btn_comport_set.width())  # 메뉴 너비를 버튼과 동일하게 설정


        # Bottom_Motor 이미지 출력
        self.bottom_motor_label = QLabel(self)
        self.bottom_motor_label.setGeometry(201, 724, 158, 26)
        self.bottom_motor_label.setStyleSheet("background: transparent;")

        image_path = resource_path("img/Bottom Motor.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(158, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.bottom_motor_label.setPixmap(pixmap)

    
        self.base_position_label = QLabel(self)
        self.base_position_label.setGeometry(70, 787, 147, 26)
        self.base_position_label.setStyleSheet("background: transparent;")

        image_path = resource_path("img/Base Position.png")  # 공백 없는 파일명 권장
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("[ERROR] 이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(
                147, 26,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.base_position_label.setPixmap(pixmap)

        # Bottom_Motor_Move Angle 이미지 출력
        self.move_angle_label = QLabel(self)
        self.move_angle_label.setGeometry(127,870,126,26)
        self.move_angle_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/Move Angle.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(126, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.move_angle_label.setPixmap(pixmap)

        # Bottom_Motor_input_angle
        self.b_abs = QLineEdit(self)
        self.b_abs.setGeometry(230,770,120,60)
        self.b_abs.setAlignment(Qt.AlignCenter)
        self.b_abs.setStyleSheet("""
                QLineEdit {
                    background: #FFFFFF;
                    border: 1px solid #D9D9D9;
                    border-radius: 15px;
                    font-family: 'NanumSquareOTF';
                    font-size: 24px;
                    font-weight: normal;
                    color: #000000;
                    padding-left: 10px;
                }
                QLineEdit:focus {
                    border: 2px solid #999999;
                    border-radius: 15px;
                    opacity: 1;
                }
            """)

        # Bottom_Motor_input_move_angle
        self.b_step = QLineEdit(self)
        self.b_step.setGeometry(70,910,240,60)
        self.b_step.setAlignment(Qt.AlignCenter)
        self.b_step.setStyleSheet("""
                QLineEdit {
                    background: #FFFFFF;
                    border: 1px solid #D9D9D9;
                    border-radius: 15px;
                    font-family: 'NanumSquareOTF';
                    font-size: 24px;
                    font-weight: normal;
                    color: #000000;
                    padding-left: 10px;
                }
                QLineEdit:focus {
                    border: 2px solid #999999;
                    border-radius: 15px;
                    opacity: 1;
                }                  
            """)
        
        # Bottom_Motor_input_move_angle2
        self.b2_step = QLineEdit(self)
        self.b2_step.setGeometry(70,990,240,60)
        self.b2_step.setAlignment(Qt.AlignCenter)
        self.b2_step.setStyleSheet("""
                QLineEdit {
                    background: #FFFFFF;
                    border: 1px solid #D9D9D9;
                    border-radius: 15px;
                    font-family: 'NanumSquareOTF';
                    font-size: 24px;
                    font-weight: normal;
                    color: #000000;
                    padding-left: 10px;
                }
                QLineEdit:focus {
                    border: 2px solid #999999;
                    border-radius: 15px;
                    opacity: 1;
                }                  
            """)
        
        # Bottom_Motor_angle_save 버튼
        self.rect108_btn = QPushButton(self)
        self.rect108_btn.setGeometry(370,770,120,60)
        BM_AS = resource_path("img/구성 요소 2 – 2.png").replace("\\", "/")
        BM_AS2 = resource_path("img/구성 요소 2 – 3.png").replace("\\", "/")
        self.rect108_btn.setStyleSheet(f"""
            QPushButton {{
                border: solid #515F70;
                border-radius: 15px;
                background: transparent url('{BM_AS}');            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{BM_AS2}');
            }}
        """)
        self.rect108_btn.clicked.connect(lambda: self.sendAbsolute(1))


        # Bottom_Motor_left 버튼
        self.bottom_rect111_left_btn = QPushButton(self)
        self.bottom_rect111_left_btn.setGeometry(331,910,160,60)
        BM_left = resource_path("img/구성 요소 3 – 1.png").replace("\\", "/")
        BM_left2 = resource_path("img/구성 요소 3 – 1.png").replace("\\", "/")
        self.bottom_rect111_left_btn.setStyleSheet(f"""
            QPushButton {{
                border: solid #515F70;
                border-radius: 15px;
                background: transparent url('{BM_left}');
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{BM_left2}');
                border: solid #515F70;
                color: #515F70;
            }}
        """)
        self.bottom_rect111_left_btn.clicked.connect(self.bottom_left_click)


        # # Bottom_Motor_right 버튼
        self.rect112_btn = QPushButton(self)
        self.rect112_btn.setGeometry(330,990,160,60)
        BM_right = resource_path("img/구성 요소 4 – 1.png").replace("\\", "/")
        BM_right2 = resource_path("img/구성 요소 4 – 2.png").replace("\\", "/")
        self.rect112_btn.setStyleSheet(f"""
            QPushButton {{
                border: solid #515F70;
                border-radius: 15px;
                background: transparent url('{BM_right}');
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{BM_right2}');
                border: solid #515F70;
                color: #515F70;
            }}
        """)
        self.rect112_btn.clicked.connect(self.bottom_right_click)


        # Top_Motor 이미지 출력
        self.top_motor_label = QLabel(self)
        self.top_motor_label.setGeometry(684, 724, 114, 26)
        self.top_motor_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/Top Motor.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(114, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.top_motor_label.setPixmap(pixmap)


        self.top_angle_label = QLabel(self)
        self.top_angle_label.setGeometry(530,787,147,26)
        self.top_angle_label.setStyleSheet("background: transparent;")

        image_path = resource_path("img/Base Position.png")  # 공백 없는 파일명 권장
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("[ERROR] 이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(
                147, 26,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.top_angle_label.setPixmap(pixmap)


        
        # Top_Motor_Move Angle 이미지 출력
        self.top_move_angle_label = QLabel(self)
        self.top_move_angle_label.setGeometry(587,870,126,26)
        self.top_move_angle_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/Move Angle.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(126, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.top_move_angle_label.setPixmap(pixmap)


        # Top_Motor_input_angle
        self.t_abs = QLineEdit(self)
        self.t_abs.setGeometry(690,770,120,60)
        self.t_abs.setAlignment(Qt.AlignCenter)
        self.t_abs.setStyleSheet("""
                QLineEdit {
                    background: #FFFFFF;
                    border: 1px solid #D9D9D9;
                    border-radius: 15px;
                    font-family: 'NanumSquareOTF';
                    font-size: 24px;
                    font-weight: normal;
                    color: #000000;
                    padding-left: 10px;
                }
                QLineEdit:focus {
                    border: 2px solid #999999;
                    border-radius: 15px;
                    opacity: 1;
                }                
            """)
        
        # Top_Motor_input_move_angle
        self.t_step = QLineEdit(self)
        self.t_step.setGeometry(530,910,240,60)
        self.t_step.setAlignment(Qt.AlignCenter)
        self.t_step.setStyleSheet("""
                QLineEdit {
                    background: #FFFFFF;
                    border: 1px solid #D9D9D9;
                    border-radius: 15px;
                    font-family: 'NanumSquareOTF';
                    font-size: 24px;
                    font-weight: normal;
                    color: #000000;
                    padding-left: 10px;
                }
                QLineEdit:focus {
                    border: 2px solid #999999;
                    border-radius: 15px;
                    opacity: 1;
                }                  
            """)
        

        # Top_Motor_input_move_angle2
        self.t2_step = QLineEdit(self)
        self.t2_step.setGeometry(530,990,240,60)
        self.t2_step.setAlignment(Qt.AlignCenter)
        self.t2_step.setStyleSheet("""
                QLineEdit {
                    background: #FFFFFF;
                    border: 1px solid #D9D9D9;
                    border-radius: 15px;
                    font-family: 'NanumSquareOTF';
                    font-size: 24px;
                    font-weight: normal;
                    color: #000000;
                    padding-left: 10px;
                }
                QLineEdit:focus {
                    border: 2px solid #999999;
                    border-radius: 15px;
                    opacity: 1;
                }                  
            """)
        

        # Top_Motor_angle_save 버튼
        self.top_rect108_btn = QPushButton(self)
        self.top_rect108_btn.setGeometry(830,770,120,60)
        TM_AS = resource_path("img/구성 요소 2 – 2.png").replace("\\", "/")
        TM_AS2 = resource_path("img/구성 요소 2 – 3.png").replace("\\", "/")
        self.top_rect108_btn.setStyleSheet(f"""
            QPushButton {{
                border: solid #515F70;
                border-radius: 15px;
                background: transparent url('{TM_AS}');
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{TM_AS2}');
            }}
        """)
        self.top_rect108_btn.clicked.connect(lambda: self.sendAbsolute(2))

        
        # Top_Motor_left 버튼
        self.top_rect111_left_btn = QPushButton(self)
        self.top_rect111_left_btn.setGeometry(790,910,160,60)
        TM_left = resource_path("img/구성 요소 3 – 1.png").replace("\\", "/")
        TM_left2 = resource_path("img/구성 요소 3 – 2.png").replace("\\", "/")
        self.top_rect111_left_btn.setStyleSheet(f"""
            QPushButton {{
                border: solid #515F70;
                border-radius: 15px;
                background: transparent url('{TM_left}');
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{TM_left2}');
                border: solid #515F70;
            }}
        """)
        self.top_rect111_left_btn.clicked.connect(self.top_left_click)

        
        # Top_Motor_right 버튼
        self.top_rect112_btn = QPushButton(self)
        self.top_rect112_btn.setGeometry(790,990,160,60)
        TM_right = resource_path("img/구성 요소 4 – 1.png").replace("\\", "/")
        TM_right2 = resource_path("img/구성 요소 4 – 2.png").replace("\\", "/")
        self.top_rect112_btn.setStyleSheet(f"""
            QPushButton {{
                border: solid #515F70;
                border-radius: 15px;
                background: transparent url('{TM_right}');
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{TM_right2}');
                border: solid #515F70;
            }}
        """)
        self.top_rect112_btn.clicked.connect(self.top_right_click)


        self.log_line1 = QLabel(self)
        self.log_line1.setGeometry(70,601,326,1)
        self.log_line1.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")

        self.log_line2 = QLabel(self)
        self.log_line2.setGeometry(624,601,326,1)
        self.log_line2.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")

        self.log_line3 = QLabel(self)
        self.log_line3.setGeometry(70,737,111,1)
        self.log_line3.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")

        self.log_line4 = QLabel(self)
        self.log_line4.setGeometry(379,737,285,1)
        self.log_line4.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")

        self.log_line5 = QLabel(self)
        self.log_line5.setGeometry(510,737,1,313)
        self.log_line5.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")

        self.log_line6 = QLabel(self)
        self.log_line6.setGeometry(818,737,132,1)
        self.log_line6.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")

        self.log_line7 = QLabel(self)
        self.log_line7.setGeometry(990,737,305,1)
        self.log_line7.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")

        self.log_line8 = QLabel(self)
        self.log_line8.setGeometry(1545,737,305,1)
        self.log_line8.setStyleSheet("border: 1px solid #CCCCCC; opacity: 1;")


        # Coordinate 라벨
        self.coordinate_label = QLabel(self)
        self.coordinate_label.setGeometry(1355, 724, 130, 26)
        self.coordinate_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/Coordinate.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(130, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.coordinate_label.setPixmap(pixmap)


        # X사각형 63 라벨
        self.rect63_label = QLabel(self)
        self.rect63_label.setGeometry(990,770,180,40)

        self.rect63_label.setStyleSheet("""
            background: transparent;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
        """)
        image_path = resource_path("img/그룹 72.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 40, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect63_label.setPixmap(pixmap)


        # X사각형 51 라벨 (하단 박스)
        self.rect51_label = QLabel(self)
        self.rect51_label.setGeometry(990, 810, 180, 60)
        self.rect51_label.setStyleSheet("""
            background: transparent;
            border: solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        image_path = resource_path("img/사각형 51.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 60, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect51_label.setPixmap(pixmap)
        

        # Y 사각형 63 라벨
        self.rect63_label = QLabel(self)
        self.rect63_label.setGeometry(1217,770,180,40)
        self.rect63_label.setStyleSheet("""
            background: transparent;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
        """)
        image_path = resource_path("img/그룹 73.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 40, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect63_label.setPixmap(pixmap)


        # Y 사각형 51 라벨 (하단 박스)
        self.rect51_label = QLabel(self)
        self.rect51_label.setGeometry(1217,810,180,60)
        self.rect51_label.setStyleSheet("""
            background: transparent;
            border: solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)

        image_path = resource_path("img/사각형 51.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 60, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect51_label.setPixmap(pixmap)

        # D 사각형 63 라벨
        self.rect63_label = QLabel(self)
        self.rect63_label.setGeometry(1443,770,180,40)

        self.rect63_label.setStyleSheet("""
            background: transparent;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
        """)

        image_path = resource_path("img/그룹 74.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 40, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect63_label.setPixmap(pixmap)

        # D 사각형 51 라벨 (하단 박스)
        self.rect51_label = QLabel(self)
        self.rect51_label.setGeometry(1443,810,180,60)

        self.rect51_label.setStyleSheet("""
            background: transparent;
            border: solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)

        image_path = resource_path("img/사각형 51.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 60, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect51_label.setPixmap(pixmap)



        # β 사각형 63 라벨
        self.rect63_label = QLabel(self)
        self.rect63_label.setGeometry(1670,770,180,40)

        self.rect63_label.setStyleSheet("""
            background: transparent;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
        """)

        image_path = resource_path("img/그룹 75.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 40, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect63_label.setPixmap(pixmap)

        # β 사각형 51 라벨 (하단 박스)
        self.rect51_label = QLabel(self)
        self.rect51_label.setGeometry(1670,810,180,60)

        self.rect51_label.setStyleSheet("""
            background: transparent;
            border: solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)

        image_path = resource_path("img/사각형 51.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(180, 60, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.rect51_label.setPixmap(pixmap)

        

        # X
        self.coord_x_label = QLabel(self)
        self.coord_x_label.setGeometry(1033, 827, 104, 26)
        self.coord_x_label.setStyleSheet("font-size: 25px; color: black; background-color: #FFFFFF;")
        self.coord_x_label.setAlignment(Qt.AlignCenter)

        # Y
        self.coord_y_label = QLabel(self)
        self.coord_y_label.setGeometry(1260, 827, 104, 26)
        self.coord_y_label.setStyleSheet("font-size: 25px; color: black; background-color: #FFFFFF;")
        self.coord_y_label.setAlignment(Qt.AlignCenter)

        # d (거리)
        self.coord_d_label = QLabel(self)
        self.coord_d_label.setGeometry(1486, 827, 104, 26)
        self.coord_d_label.setStyleSheet("font-size: 25px; color: black; background-color: #FFFFFF;")
        self.coord_d_label.setAlignment(Qt.AlignCenter)

        # β (각도)
        self.coord_b_label = QLabel(self)
        self.coord_b_label.setGeometry(1721, 827, 100, 26)
        self.coord_b_label.setStyleSheet("font-size: 25px; color: black; background-color: #FFFFFF;")
        self.coord_b_label.setAlignment(Qt.AlignCenter)


        self.lidar_canvas.coord_signal.connect(self.update_coords)


        # Raw Data Output 라벨
        self.raw_output_label = QLabel(self)
        self.raw_output_label.setGeometry(1105, 927, 190, 26)
        self.raw_output_label.setStyleSheet("background: transparent;")

        image_path = resource_path("img/Raw Data Output.png")
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print("이미지 로드 실패:", image_path)
        else:
            pixmap = pixmap.scaled(190, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.raw_output_label.setPixmap(pixmap)

        # Data Save 푸쉬버튼
        self.rect54_btn = QPushButton(self)
        self.rect54_btn.setCheckable(True)
        self.rect54_btn.setGeometry(1440,910,410,60)
        TM_right = resource_path("img/구성 요소 8 – 1.png").replace("\\", "/")
        TM_right2 = resource_path("img/구성 요소 8 – 2.png").replace("\\", "/")
        self.rect54_btn.setStyleSheet(f"""
            QPushButton {{
                border: solid #1C426D;
                border-radius: 15px;
                background: transparent url('{TM_right}') 0% 0% no-repeat;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            
            QPushButton:checked {{
                background: transparent url('{TM_right2}') 0% 0% no-repeat;
            }}
        """)
        self.rect54_btn.setCursor(Qt.PointingHandCursor)
        self.rect54_btn.clicked.connect(self.toggle_data_save)
        self.camera_thread = None

        
        # Start 버튼
        self.rect66_btn = QPushButton(self)
        self.rect66_btn.setGeometry(990, 990, 410, 60)
        TM_start = resource_path("img/그룹 66.png").replace("\\", "/")
        TM_start2 = resource_path("img/구성 요소 7 – 1.png").replace("\\", "/")
        self.rect66_btn.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-radius: 15px;
                background: transparent url('{TM_start}') 0% 0% no-repeat;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{TM_start2}');
                border: solid #1C426D;
            }}                          
        """)

        self.rect66_btn.setCursor(Qt.PointingHandCursor)
        self.rect66_btn.clicked.connect(self.handle_start)


        # END 버튼
        self.rect67_btn = QPushButton(self)
        self.rect67_btn.setGeometry(1440,990,410,60)
        TM_end = resource_path("img/구성 요소 10 – 1.png").replace("\\", "/")
        TM_end2 = resource_path("img/구성 요소 10 – 2.png").replace("\\", "/")
        self.rect67_btn.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-radius: 15px;
                background: transparent url('{TM_end}') 0% 0% no-repeat;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                background: transparent url('{TM_end2}');
                border: solid #1C426D;
            }}                          
        """)

        self.rect67_btn.setCursor(Qt.PointingHandCursor)
        self.rect67_btn.clicked.connect(self.handle_end)


        # 실행 시 전체화면
        self.showFullScreen()


        # RTSP URL
        self.rtsp_url = "rtsp://admin:ajwptm12!@192.168.0.64:554/Streaming/Channels/101"
    
        # 카메라 스레드 생성
        self.cam_thread = CameraThread(
            ui_ref=self,
            rtsp_url=self.rtsp_url,
            out_dir=OUT_DIR,
            fourcc="mp4v",
            target_fps=30,
            yolo_model_path="model/best.pt",  # ★★★ 가중치 파일 경로
            enable_yolo=True             # ★★★ YOLO 활성화
        )

        # UI 업데이트 연결
        self.cam_thread.frame_signal.connect(self.update_camera_frame)
        # self.cam_thread.fps_signal.connect(self.update_fps_label)

        # 스레드 시작
        self.cam_thread.start()

    # pressed 상태에서 overlay 표시 여부 제어
    def show_overlay_on_press(self):
        if self.rect54_btn.isChecked():
            # 이미 SAVE ON 상태 > overlay를 띄우지 않음
            self.data_save_overlay.hide()
        else:
            # SAVE OFF 상태 > overlay 표시
            self.data_save_overlay.show()

    

    def toggle_data_save(self):

        btn = self.rect54_btn

        # SYSTEM OFF
        if not self.state_icon.system_started:
            QMessageBox.warning(
                self,
                "SYSTEM",
                "State ON 상태에서만 저장이 가능합니다."
            )
            # 버튼 상태 원래대로 되돌리기
            btn.setChecked(False)
            return

        
        # SAVE ON (OFF > ON)
        
        if btn.isChecked():
            QMessageBox.information(self, "SYSTEM", "Data Save ON")

            # CAMERA RECORD START
            if not self.cam_thread.recording:
                self.cam_thread.start_recording()

            # LIDAR CSV START
            canvas = self.lidar_canvas
            if not canvas.lidar_save_enabled:
                start_ts = time.strftime("%Y%m%d_%H%M%S")
                csv_filename = f"lidar_{start_ts}.csv"

                canvas.csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
                canvas.csv_writer = csv.writer(canvas.csv_file)
                canvas.csv_writer.writerow(["timestamp", "P1_x", "P1_y"])
                canvas.csv_segment_start = time.time()
                canvas.lidar_save_enabled = True
                print("[LIDAR CSV] 저장 ON")

        
        # SAVE OFF (ON > OFF)
        else:
            QMessageBox.information(self, "SYSTEM", "Data Save STOP")

            # CAMERA RECORD STOP
            if self.cam_thread.recording:
                self.cam_thread.stop_recording()

            # LIDAR CSV STOP
            canvas = self.lidar_canvas
            if canvas.lidar_save_enabled:
                canvas.lidar_save_enabled = False
                if canvas.csv_file:
                    canvas.csv_file.close()
                print("[LIDAR CSV] 저장 OFF")


    #Comport set
    def connect_serial_port(self, port_name):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass

        try:
            self.ser = serial.Serial(port_name, 115200, timeout=0.1)
            time.sleep(0.1)

            print(f"[SERIAL] Connected > {port_name}")
            self.btn_comport_set.setText(port_name)

            # 상태만 true로 — 아이콘은 건드리지 않음
            self.state_icon.comport_connected = True

        except Exception as e:
            print(f"[ERROR] Connect failed: {e}")
            self.ser = None

            # 상태만 false로 — 아이콘은 건드리지 않음
            self.state_icon.comport_connected = False


    def populate_comport_menu(self):
        self.comport_menu.clear()
        import serial.tools.list_ports
        
        ports = [p.device for p in serial.tools.list_ports.comports()]
        
        if not ports:
            self.comport_menu.addAction("No Ports Found")
            return

        for port in ports:
            action = self.comport_menu.addAction(port)
            action.triggered.connect(lambda checked, p=port: self.connect_serial_port(p))

    def on_port_selected(self, port_name):
        self.selected_port = port_name
        self.btn_comport_set.setText(port_name)  # 버튼 텍스트 변경
        print(f"[COM] 선택된 포트: {port_name}")

    def sendCmd(self, cmd):
        if not hasattr(self, 'ser') or self.ser is None:
            print("[ERROR] 포트 미연결")
            return
        self.ser.write((cmd + "\n").encode())
        print(f"[SERIAL] {cmd}")


    def sendAbsolute(self, motor):
        # motor=1 > bottom / motor=2 > top
        box = self.b_abs if motor == 1 else self.t_abs
        val = box.text().strip()

        if not val.isdigit():
            print("[ERROR] 절대각 입력 오류")
            return

        cmd = f"s{motor}:{val}"
        self.sendCmd(cmd)

    # step 설정
    def sendStep(self, motor):
        box = self.b_step if motor == 1 else self.t_step
        val = box.text().strip()

        if not val.isdigit():
            print("[ERROR] Move Angle(step) 입력 오류")
            return

        cmd = f"step{motor}:{val}"
        self.sendCmd(cmd)

    def bottom_left_click(self):
        # 먼저 step1 값 설정
        val = self.b_step.text().strip()
        if val.isdigit():
            self.sendCmd(f"step1:{val}")
        else:
            print("[ERROR] Bottom MoveAngle 입력 오류")
            return

        # 이제 LEFT 이동
        self.sendCmd("s1:left")

    def bottom_right_click(self):
        # b2_step 에 입력된 값을 가져옴
        val = self.b2_step.text().strip()

        # step1 설정
        if val.isdigit():
            self.sendCmd(f"step1:{val}")
        else:
            print("[ERROR] Bottom MoveAngle2 입력 오류")
            return

        # right 이동
        self.sendCmd("s1:right")

    def top_left_click(self):
        # 입력값 읽기
        val = self.t_step.text().strip()

        # step2 설정
        if val.isdigit():
            self.sendCmd(f"step2:{val}")
        else:
            print("[ERROR] Top MoveAngle(step) 입력 오류")
            return

        # left 이동 명령
        self.sendCmd("s2:left")

    def top_right_click(self):
        # 입력값 읽기
        val = self.t2_step.text().strip()

        # step2 설정
        if val.isdigit():
            self.sendCmd(f"step2:{val}")
        else:
            print("[ERROR] Top MoveAngle2(step) 입력 오류")
            return

        # right 이동
        self.sendCmd("s2:right")

    def check_servo_device(self, timeout=1.0):
        try:
            # 입력 버퍼 비우기 (중요)
            self.ser.reset_input_buffer()

            # ping 전송
            self.ser.write(b"ping\n")

            start = time.time()
            while time.time() - start < timeout:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    print("[SERVO CHECK]", line)

                    if line == "SERVO_OK":
                        return True

            return False

        except Exception as e:
            print("[SERVO CHECK ERROR]", e)
            return False



    def handle_start(self):

        # 1️⃣ 카메라 / LiDAR / COM 기본 연결 체크
        if not self.state_icon.camera_connected:
            QMessageBox.warning(self, "SYSTEM", "카메라가 연결되지 않았습니다.")
            return

        if not self.state_icon.lidar_connected:
            QMessageBox.warning(self, "SYSTEM", "LiDAR가 연결되지 않았습니다.")
            return

        if not self.state_icon.comport_connected:
            QMessageBox.warning(self, "SYSTEM", "서보모터 COM 포트가 연결되지 않았습니다.")
            return

        # 2️⃣ 서보모터 장치 검증 (핵심)
        if not self.check_servo_device():
            QMessageBox.critical(
                self,
                "SYSTEM ERROR",
                "선택한 COM 포트는 서보모터가 아닙니다.\n"
                "올바른 장치를 선택해주세요."
            )
            
            self.state_icon.system_started = False
            self.state_icon.update_state()
            return

        # 3️⃣ 모든 검증 통과 → SYSTEM ON
        self.state_icon.system_started = True
        self.state_icon.update_state()
        self.log_timer.start(2000)

        print("[SYSTEM] START ON > 모든 장치 정상")
        QMessageBox.information(self, "SYSTEM", "시스템이 활성화되었습니다.")



    # def handle_start(self):
    #     # 3가지 센서 상태 검사
    #     if not (self.state_icon.camera_connected and
    #             self.state_icon.lidar_connected and
    #             self.state_icon.comport_connected):

    #         QMessageBox.warning(
    #             self,
    #             "장치 연결 오류",
    #             "카메라, LiDAR, COMPORT가 모두 연결되어야 시작 가능합니다."
    #         )
    #         return

    #     # 여기까지 통과 > STATE GREEN
    #     self.state_icon.system_started = True
    #     self.state_icon.update_state()

    #     print("[SYSTEM] START ON > 모든 장치 정상 연결")
    #     QMessageBox.information(self, "SYSTEM", "시스템이 활성화되었습니다.")


    def handle_end(self):
        # [추가] 5초 주기 로그 타이머 중지
        if hasattr(self, 'log_timer'):
            self.log_timer.stop()
            
        # 시스템 비활성화
        self.state_icon.system_started = False
        self.state_icon.update_state()
    

        # 시리얼 종료
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.selected_port = None
        self.btn_comport_set.setText("COMPORT SET")

        # 카메라 녹화 중단
        if self.cam_thread.recording:
            self.cam_thread.stop_recording()

        # LiDAR CSV 중단
        canvas = self.lidar_canvas
        if canvas.lidar_save_enabled:
            canvas.lidar_save_enabled = False
            if canvas.csv_file:
                canvas.csv_file.close()

        # 상태 아이콘 RED로 초기화
        self.state_icon.reset_all()

        print("[SYSTEM] END 완료")
        QMessageBox.information(self, "SYSTEM", "시스템이 비활성화되었습니다.")

    def update_timestamp(self):
        now = QDateTime.currentDateTime()
        self.log_timestamp.setText(now.toString("yyyy/MM/dd - HH:mm"))

    def check_device_status(self):
        now = time.time()

        # 카메라 상태 확인
        camera_alive = (now - self.last_camera_frame_time < 15.0)
        lidar_alive  = (now - self.last_lidar_frame_time  < 5.0)

        self.state_icon.camera_connected = camera_alive
        self.state_icon.lidar_connected  = lidar_alive

        # 🔥 START 승인 이후에만 상태 반영
        if self.state_icon.system_started:
            self.state_icon.update_state()

        
        #  카메라 자동 재연결 로직 추가 위치
        
        if not camera_alive:
            print("[CAM] 끊김 감지 > 자동 재연결 시도")

            try:
                self.cam_thread.stop()
                time.sleep(1.0)

                # CameraThread 재생성
                self.cam_thread = CameraThread(
                    self,
                    self.rtsp_url,
                    OUT_DIR,
                    "mp4v",
                    30,
                    yolo_model_path='model/best.pt',
                    enable_yolo=True
                )
                self.cam_thread.frame_signal.connect(self.update_camera_frame)
                self.cam_thread.start()

                # [중요] 재연결 시도했으므로 타이머 리셋 (연속 재부팅 방지)
                self.last_camera_frame_time = time.time()
                print("[CAM] 자동 재연결 성공")

            except Exception as e:
                print("[CAM] 자동 재연결 실패:", e)

        #  LiDAR 자동 재연결 로직도 바로 아래에 위치
        if not lidar_alive:
            print("[LIDAR] 끊김 감지 > 자동 재연결 시도")
            try:
                global sock
                sock.close()

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(("192.168.0.99", 2111))
                sock.sendall(b"\x02sEN LMDscandata 1\x03")

                print("[LIDAR] 자동 재연결 성공")

                self.last_lidar_frame_time = time.time()
                self.state_icon.lidar_connected = True
            except:
                print("[LIDAR] 자동 재연결 실패")


if __name__ == "__main__":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|"
        "stimeout;10000000|"
        "max_delay;5000000|"
        "buffer_size;1048576"
    )
    app = QApplication(sys.argv)
    ui = MyUI()
    ui.show()
    sys.exit(app.exec_())
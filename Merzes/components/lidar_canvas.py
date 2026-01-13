import time
import csv
import math
import numpy as np
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
from matplotlib.ticker import MultipleLocator, AutoLocator

import config
from hardware.lidar import LidarConnection, parse_lmd, cluster_and_classify


class LidarCanvas(FigureCanvas):
    """LiDAR 데이터 시각화 캔버스"""
    
    coord_signal = pyqtSignal(float, float, float, float)
    
    def __init__(self, ui_ref, lidar_conn, parent=None):
        self.fig = Figure(figsize=(8.6, 5.4), dpi=100)
        super().__init__(self.fig)
        
        self.ui = ui_ref
        self.lidar_conn = lidar_conn
        self.setParent(ui_ref)

        # CSV 저장 관련
        self.lidar_save_enabled = False
        self.csv_file = None
        self.csv_writer = None
        self.csv_segment_sec = config.LIDAR_CSV_SEGMENT_SEC
        self.csv_segment_start = None

        # 회전 보정
        self.ANGLE_OFFSET = np.deg2rad(config.LIDAR_ANGLE_OFFSET)

        # 플롯 설정
        self._setup_plot()
        
        # 마우스 이벤트 연결
        self._connect_mouse_events()
        
        # 애니메이션 시작
        self.ani = FuncAnimation(
            self.fig, 
            self.update_lidar, 
            interval=config.LIDAR_UPDATE_INTERVAL, 
            blit=False, 
            cache_frame_data=False
        )

    def _setup_plot(self):
        """플롯 초기 설정"""
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-config.LIDAR_MAX_RANGE, config.LIDAR_MAX_RANGE)
        self.ax.set_ylim(-config.LIDAR_MAX_RANGE, config.LIDAR_MAX_RANGE)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.xaxis.set_major_locator(MultipleLocator(5))
        self.ax.yaxis.set_major_locator(MultipleLocator(5))
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_title("TiM571", pad=20)
        self.ax.xaxis.set_ticks_position("top")
        self.ax.xaxis.set_label_position("top")

        # 투명 배경
        self.setStyleSheet("background: transparent;")
        self.fig.patch.set_facecolor("none")
        self.ax.set_facecolor("none")
        
        # FOV 표시
        fov = Wedge(
            (0, 0), 
            config.LIDAR_MAX_RANGE, 
            config.LIDAR_FOV_START, 
            config.LIDAR_FOV_END, 
            facecolor="lightgray", 
            alpha=0.3
        )
        self.ax.add_patch(fov)

        # LiDAR 위치 마커
        self.ax.plot(0, 0, "ks", markersize=8)

        # 산점도 초기화
        self.scat = self.ax.scatter([], [], s=8)

    def _connect_mouse_events(self):
        """마우스 이벤트 연결"""
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def on_mouse_move(self, event):
        """마우스 이동 이벤트"""
        if event.inaxes is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = event.xdata
        y = event.ydata
        d = (x**2 + y**2)**0.5
        beta = np.degrees(np.arctan2(y, x))

        self.coord_signal.emit(x, y, d, beta)

    def on_scroll(self, event):
        """마우스 휠 확대/축소"""
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
        """마우스 버튼 누름"""
        if event.xdata is None:
            return
        self._pan = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, event):
        """마우스 드래그"""
        if not hasattr(self, "_pan") or event.inaxes != self.ax:
            return
        x0, y0, (x1, x2), (y1, y2) = self._pan
        dx = event.xdata - x0
        dy = event.ydata - y0
        self.ax.set_xlim(x1 - dx, x2 - dx)
        self.ax.set_ylim(y1 - dy, y2 - dy)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        """마우스 버튼 릴리즈"""
        if hasattr(self, "_pan"):
            del self._pan
    
    def update_lidar(self, _):
        """LiDAR 데이터 업데이트"""
        # 데이터 수신
        msg = self.lidar_conn.receive()
        
        if msg is None:
            self.ui.state_icon.lidar_connected = False
            return self.scat,

        # 파싱
        th, r = parse_lmd(msg)
        if th is None:
            self.ui.state_icon.lidar_connected = False
            return self.scat,

        # 상태 업데이트
        self.ui.state_icon.lidar_connected = True
        self.ui.last_lidar_frame_time = time.time()

        # 극좌표 -> 직교좌표 변환
        th = (th + self.ANGLE_OFFSET) % (2 * np.pi)
        x = -r * np.sin(th)
        y = r * np.cos(th)
        pts = np.column_stack((x, y))

        # 클러스터링 및 분류
        colors, person_centroids = cluster_and_classify(pts)

        # 산점도 업데이트
        self.scat.set_offsets(pts)
        self.scat.set_color(colors)

        # 추적 시스템에 사람 위치 전달
        if hasattr(self.ui, 'tracking_system'):
            self.ui.tracking_system.update_target(person_centroids)

        # CSV 저장
        if self.lidar_save_enabled:
            self._save_to_csv(person_centroids)

        return self.scat,

    def _save_to_csv(self, person_centroids):
        """CSV 파일에 저장"""
        # 분 단위 파일 분할
        if time.time() - self.csv_segment_start >= self.csv_segment_sec:
            if self.csv_file:
                self.csv_file.close()

            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_filename = f"lidar_{ts}.csv"
            self.csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["timestamp", "P1_x", "P1_y"])

            self.csv_segment_start = time.time()
            print(f"[LIDAR CSV] 새 파일 생성: {csv_filename}")

        # 데이터 기록
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        num_p = len(person_centroids)

        if num_p == 0:
            self.csv_writer.writerow([ts, "NONE"])
        else:
            coord_flat = []
            for (px, py) in person_centroids:
                coord_flat.append(f"{px:.3f}")
                coord_flat.append(f"{py:.3f}")

            # 사람 간 거리 계산
            dist_list = []
            for i in range(num_p):
                for j in range(i + 1, num_p):
                    dx = person_centroids[i][0] - person_centroids[j][0]
                    dy = person_centroids[i][1] - person_centroids[j][1]
                    dist_cm = math.sqrt(dx * dx + dy * dy) * 100
                    dist_list.append(f"(p{i+1}-p{j+1}){dist_cm:.2f}")

            self.csv_writer.writerow([ts] + coord_flat + dist_list)

    def start_csv_recording(self):
        """CSV 녹화 시작"""
        if self.lidar_save_enabled:
            return

        start_ts = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"lidar_{start_ts}.csv"

        self.csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "P1_x", "P1_y"])
        self.csv_segment_start = time.time()
        self.lidar_save_enabled = True
        print("[LIDAR CSV] 저장 ON")

    def stop_csv_recording(self):
        """CSV 녹화 중지"""
        if not self.lidar_save_enabled:
            return

        self.lidar_save_enabled = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        print("[LIDAR CSV] 저장 OFF")
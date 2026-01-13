import time
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QPixmap, QKeySequence, QPainter, QPainterPath
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QMessageBox, QShortcut, QPushButton

import config
from utils.resource_utils import resource_path
from utils.serial_utils import get_available_ports, connect_serial, send_command, check_servo_device
from hardware.camera import CameraThread
from hardware.lidar import LidarConnection
from components.state_manager import StateManager
from components.lidar_canvas import LidarCanvas
from components.tracking_system import TrackingSystem
from .widgets import RoundedMenu, ToggleButton


class MainWindow(QWidget):
    """메인 윈도우 클래스"""
    
    def __init__(self):
        super().__init__()
        
        # 상태 관리자
        self.state_icon = StateManager(self)
        
        # 시리얼 통신
        self.ser = None
        self.selected_port = None
        
        # 디바이스 상태
        self.last_camera_frame_time = 0
        self.last_lidar_frame_time = 0
        
        # LiDAR 연결
        self.lidar_conn = LidarConnection()
        
        # 메뉴 상태
        self.menu_open = False
        
        # 추적 시스템 초기화
        self.tracking_system = TrackingSystem(self)
        self._connect_tracking_signals()
        
        # UI 초기화
        self.initUI()
        
        # 타이머 설정
        self._setup_timers()
        
        # 카메라 스레드 시작
        self._start_camera_thread()
    
    def _connect_tracking_signals(self):
        """추적 시스템 시그널 연결"""
        self.tracking_system.target_detected.connect(self._on_target_detected)
        self.tracking_system.tracking_started.connect(self._on_tracking_started)
        self.tracking_system.tracking_stopped.connect(self._on_tracking_stopped)
        self.tracking_system.motor_moved.connect(self._on_motor_moved)
    
    def _on_target_detected(self, x, y):
        """타겟 감지 시 호출"""
        print(f"[TARGET] 감지: ({x:.2f}, {y:.2f})")
    
    def _on_tracking_started(self):
        """추적 시작 시 호출"""
        print("[TRACKING] 자동 추적 시작됨")
    
    def _on_tracking_stopped(self):
        """추적 중지 시 호출"""
        print("[TRACKING] 자동 추적 중지됨")
    
    def _on_motor_moved(self, motor_id, angle):
        """모터 이동 시 호출"""
        motor_name = "Bottom" if motor_id == 1 else "Top"
        print(f"[MOTOR] {motor_name} Motor > {angle:.1f}°")
    
    def _setup_timers(self):
        """타이머 설정"""
        self.device_check_timer = QTimer()
        self.device_check_timer.timeout.connect(self.check_device_status)
        self.device_check_timer.start(config.DEVICE_CHECK_INTERVAL)
        
        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.update_timestamp)
        self.timer2.start(config.TIMESTAMP_UPDATE_INTERVAL)
        self.update_timestamp()
    
    def _start_camera_thread(self):
        """카메라 스레드 시작"""
        self.cam_thread = CameraThread(
            ui_ref=self,
            rtsp_url=config.RTSP_URL,
            out_dir=config.CAMERA_OUT_DIR,
            fourcc=config.CAMERA_FOURCC,
            target_fps=config.CAMERA_TARGET_FPS
        )
        
        self.cam_thread.frame_signal.connect(self.update_camera_frame)
        self.cam_thread.start()
    
    def initUI(self):
        """UI 초기화"""
        self.setWindowTitle("Motor_BlackBox")
        self.setGeometry(0, 0, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.setStyleSheet("background-color: #FCFBFB;")
        
        self.shortcut_fullscreen = QShortcut(QKeySequence("F11"), self)
        self.shortcut_fullscreen.activated.connect(self.toggle_fullscreen)
        
        self._create_header()
        self._create_camera_section()
        self._create_lidar_section()
        self._create_servo_controls()
        self._create_coordinate_display()
        self._create_control_buttons()
        
        self.showFullScreen()
    
    def _create_header(self):
        """헤더 영역 생성"""
        self.logo_image_label = QLabel(self)
        self.logo_image_label.setGeometry(80, 20, 260, 50)
        logo_image_path = resource_path("img/로고 02.png")
        pixmap = QPixmap(logo_image_path)
        self.logo_image_label.setPixmap(
            pixmap.scaled(260, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        
        self.state_label = QLabel(self)
        self.state_label.setGeometry(780, 37, 60, 26)
        self.state_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/State.png")
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(60, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.state_label.setPixmap(pixmap)
        
        self.state_label_circle = QLabel(self)
        self.state_label_circle.setGeometry(925, 40, 20, 20)
        self.state_label_circle.setStyleSheet("background: transparent;")
        self.state_icon.update_state()
        
        self.log_timestamp = QLabel(self)
        self.log_timestamp.setGeometry(1602, 35, 248, 30)
        self.log_timestamp.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.log_timestamp.setStyleSheet(
            "font-family: 'NanumSquareOTF'; font-size: 27px; color: #000000;"
        )
    
    def _create_camera_section(self):
        """카메라 영역 생성"""
        self.cam_label = QLabel(self)
        self.cam_label.setGeometry(70, 90, 880, 40)
        self.cam_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/그룹 77.png")
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(880, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_label.setPixmap(pixmap)
        
        self.rect32_label = QLabel(self)
        self.rect32_label.setGeometry(70, 130, 880, 420)
        self.rect32_label.setStyleSheet("""
            background: transparent;
            border: solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(70, 130, 880, 420)
        self.camera_label.setScaledContents(True)
        self.camera_label.setStyleSheet("""
            background: transparent;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        self.camera_label.raise_()
    
    def _create_lidar_section(self):
        """LiDAR 영역 생성"""
        self.map_label = QLabel(self)
        self.map_label.setGeometry(990, 90, 860, 40)
        self.map_label.setStyleSheet("background: transparent;")
        image_path = resource_path("img/그룹 76.png")
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(860, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.map_label.setPixmap(pixmap)
        
        self.rect48_label = QLabel(self)
        self.rect48_label.setGeometry(990, 130, 860, 540)
        self.rect48_label.setStyleSheet("""
            background: transparent;
            border: solid #CCCCCC;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        
        self.lidar_canvas = LidarCanvas(self, self.lidar_conn)
        self.lidar_canvas.setGeometry(990, 130, 860, 540)
        self.lidar_canvas.coord_signal.connect(self.update_coords)
        self.lidar_canvas.raise_()
    
    def _create_servo_controls(self):
        """서보모터 제어 영역 생성"""
        # (여기에 servo control 코드 - 너무 길어서 생략, 기존 코드 그대로 사용)
        pass
    
    def _create_coordinate_display(self):
        """좌표 표시 영역 생성"""
        # (여기에 coordinate display 코드 - 기존 코드 그대로 사용)
        pass
    
    def _create_control_buttons(self):
        """제어 버튼들 생성"""
        # Auto Tracking 버튼 추가
        self.auto_tracking_label = QLabel("Auto Tracking", self)
        self.auto_tracking_label.setGeometry(990, 880, 150, 26)
        self.auto_tracking_label.setStyleSheet(
            "font-family: 'NanumSquareOTF'; font-size: 20px; "
            "font-weight: bold; color: #1C426D;"
        )
        
        self.tracking_toggle_btn = QPushButton(self)
        self.tracking_toggle_btn.setCheckable(True)
        self.tracking_toggle_btn.setGeometry(1150, 870, 250, 50)
        self.tracking_toggle_btn.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                border: 2px solid #1C426D;
                border-radius: 15px;
                font-family: 'NanumSquareOTF';
                font-size: 20px;
                color: #1C426D;
                font-weight: bold;
            }
            QPushButton:hover { background: #F0F0F0; }
            QPushButton:checked { background: #1C426D; color: #FFFFFF; }
        """)
        self.tracking_toggle_btn.setText("TRACKING OFF")
        self.tracking_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.tracking_toggle_btn.clicked.connect(self.toggle_auto_tracking)
        
        # (나머지 버튼들 - 기존 코드 그대로)
    
    # UI 업데이트 메서드들
    def update_camera_frame(self, qimg):
        """카메라 프레임 업데이트"""
        pix = QPixmap.fromImage(qimg)
        rounded = QPixmap(pix.size())
        rounded.fill(Qt.transparent)
        
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)
        
        radius = 15
        path = QPainterPath()
        w, h = pix.width(), pix.height()
        
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
        self.last_camera_frame_time = time.time()
    
    def update_coords(self, x, y, d, beta):
        """좌표 업데이트"""
        self.coord_x_label.setText(f"{x:.3f}m")
        self.coord_y_label.setText(f"{y:.3f}m")
        self.coord_d_label.setText(f"{d:.3f}m")
        self.coord_b_label.setText(f"{beta:.2f}°")
    
    def update_timestamp(self):
        """타임스탬프 업데이트"""
        now = QDateTime.currentDateTime()
        self.log_timestamp.setText(now.toString("yyyy/MM/dd - HH:mm"))
    
    def toggle_fullscreen(self):
        """전체화면 토글"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    # 이벤트 핸들러들
    def toggle_auto_tracking(self):
        """자동 추적 토글"""
        btn = self.tracking_toggle_btn
        
        if not self.state_icon.system_started:
            QMessageBox.warning(self, "AUTO TRACKING", 
                "State ON 상태에서만 자동 추적이 가능합니다.")
            btn.setChecked(False)
            return
        
        if btn.isChecked():
            if self.tracking_system.start_tracking():
                btn.setText("TRACKING ON")
                QMessageBox.information(self, "AUTO TRACKING", 
                    "자동 추적이 시작되었습니다.")
            else:
                btn.setChecked(False)
                QMessageBox.warning(self, "AUTO TRACKING", 
                    "자동 추적을 시작할 수 없습니다.")
        else:
            self.tracking_system.stop_tracking()
            btn.setText("TRACKING OFF")
            QMessageBox.information(self, "AUTO TRACKING", 
                "자동 추적이 중지되었습니다.")
    
    def toggle_comport_menu(self):
        """COM 포트 메뉴 토글"""
        try:
            if self.comport_menu.isVisible():
                self.comport_menu.hide()
                return
            
            self.populate_comport_menu()
            pos = self.btn_comport_set.mapToGlobal(
                self.btn_comport_set.rect().bottomLeft()
            )
            self.comport_menu.popup(pos)
        except Exception as e:
            print("toggle error:", e)
    
    def populate_comport_menu(self):
        """COM 포트 메뉴 채우기"""
        self.comport_menu.clear()
        ports = get_available_ports()
        
        if not ports:
            self.comport_menu.addAction("No Ports Found")
            return
        
        for port in ports:
            action = self.comport_menu.addAction(port)
            action.triggered.connect(
                lambda checked, p=port: self.connect_serial_port(p)
            )
    
    def connect_serial_port(self, port_name):
        """시리얼 포트 연결"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        
        self.ser = connect_serial(
            port_name, 
            config.SERIAL_BAUDRATE, 
            config.SERIAL_TIMEOUT
        )
        
        if self.ser:
            self.btn_comport_set.setText(port_name)
            self.state_icon.comport_connected = True
        else:
            self.state_icon.comport_connected = False
    
    def _menu_closed_block(self):
        """메뉴 닫힘 처리"""
        self.menu_open = False
        self.btn_comport_set.blockSignals(True)
        QTimer.singleShot(200, lambda: self.btn_comport_set.blockSignals(False))
    
    def sendCmd(self, cmd):
        """시리얼 명령 전송"""
        return send_command(self.ser, cmd)
    
    def sendAbsolute(self, motor):
        """절대 각도 설정"""
        box = self.b_abs if motor == 1 else self.t_abs
        val = box.text().strip()
        
        if not val.isdigit():
            return
        
        self.sendCmd(f"s{motor}:{val}")
    
    def bottom_left_click(self):
        """Bottom Motor 왼쪽"""
        val = self.b_step.text().strip()
        if val.isdigit():
            self.sendCmd(f"step1:{val}")
            self.sendCmd("s1:left")
    
    def bottom_right_click(self):
        """Bottom Motor 오른쪽"""
        val = self.b2_step.text().strip()
        if val.isdigit():
            self.sendCmd(f"step1:{val}")
            self.sendCmd("s1:right")
    
    def top_left_click(self):
        """Top Motor 왼쪽"""
        val = self.t_step.text().strip()
        if val.isdigit():
            self.sendCmd(f"step2:{val}")
            self.sendCmd("s2:left")
    
    def top_right_click(self):
        """Top Motor 오른쪽"""
        val = self.t2_step.text().strip()
        if val.isdigit():
            self.sendCmd(f"step2:{val}")
            self.sendCmd("s2:right")
    
    def toggle_data_save(self):
        """데이터 저장 토글"""
        btn = self.rect54_btn
        
        if not self.state_icon.system_started:
            QMessageBox.warning(self, "SYSTEM", 
                "State ON 상태에서만 저장이 가능합니다.")
            btn.setChecked(False)
            return
        
        if btn.isChecked():
            QMessageBox.information(self, "SYSTEM", "Data Save ON")
            
            if not self.cam_thread.recording:
                self.cam_thread.start_recording()
            
            self.lidar_canvas.start_csv_recording()
        else:
            QMessageBox.information(self, "SYSTEM", "Data Save STOP")
            
            if self.cam_thread.recording:
                self.cam_thread.stop_recording()
            
            self.lidar_canvas.stop_csv_recording()
    
    def handle_start(self):
        """시스템 시작"""
        if not self.state_icon.camera_connected:
            QMessageBox.warning(self, "SYSTEM", "카메라가 연결되지 않았습니다.")
            return
        
        if not self.state_icon.lidar_connected:
            QMessageBox.warning(self, "SYSTEM", "LiDAR가 연결되지 않았습니다.")
            return
        
        if not self.state_icon.comport_connected:
            QMessageBox.warning(self, "SYSTEM", 
                "서보모터 COM 포트가 연결되지 않았습니다.")
            return
        
        if not check_servo_device(self.ser):
            QMessageBox.critical(self, "SYSTEM ERROR",
                "선택한 COM 포트는 서보모터가 아닙니다.\n"
                "올바른 장치를 선택해주세요.")
            
            self.state_icon.system_started = False
            self.state_icon.update_state()
            return
        
        self.state_icon.system_started = True
        self.state_icon.update_state()
        
        QMessageBox.information(self, "SYSTEM", "시스템이 활성화되었습니다.")
    
    def handle_end(self):
        """시스템 종료"""
        if self.tracking_system.tracking_enabled:
            self.tracking_system.stop_tracking()
            self.tracking_toggle_btn.setChecked(False)
            self.tracking_toggle_btn.setText("TRACKING OFF")
        
        self.state_icon.system_started = False
        self.state_icon.update_state()
        
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.selected_port = None
        self.btn_comport_set.setText("COMPORT SET")
        
        if self.cam_thread.recording:
            self.cam_thread.stop_recording()
        
        self.lidar_canvas.stop_csv_recording()
        self.state_icon.reset_all()
        
        QMessageBox.information(self, "SYSTEM", "시스템이 비활성화되었습니다.")
    
    def check_device_status(self):
        """디바이스 상태 주기적 확인"""
        now = time.time()
        
        camera_alive = (now - self.last_camera_frame_time < config.DEVICE_TIMEOUT)
        lidar_alive = (now - self.last_lidar_frame_time < config.DEVICE_TIMEOUT)
        
        self.state_icon.camera_connected = camera_alive
        self.state_icon.lidar_connected = lidar_alive
        
        if self.state_icon.system_started:
            self.state_icon.update_state()
        
        if not camera_alive:
            try:
                self.cam_thread.stop()
                time.sleep(0.5)
                
                self.cam_thread = CameraThread(
                    self, config.RTSP_URL, config.CAMERA_OUT_DIR,
                    config.CAMERA_FOURCC, config.CAMERA_TARGET_FPS
                )
                self.cam_thread.frame_signal.connect(self.update_camera_frame)
                self.cam_thread.start()
            except Exception as e:
                print("[CAM] 재연결 실패:", e)
        
        if not lidar_alive:
            try:
                self.lidar_conn.reconnect()
                self.last_lidar_frame_time = time.time()
                self.state_icon.lidar_connected = True
            except:
                pass
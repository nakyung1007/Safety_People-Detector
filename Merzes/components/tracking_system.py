"""
LiDAR-Camera 통합 추적 시스템
LiDAR에서 감지된 사람 위치로 카메라를 자동 이동
"""
import time
import math
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from utils.serial_utils import send_command
import config


class TrackingSystem(QObject):
    """LiDAR-Camera 통합 추적 시스템"""
    
    # 시그널 정의
    target_detected = pyqtSignal(float, float)  # x, y 좌표
    tracking_started = pyqtSignal()
    tracking_stopped = pyqtSignal()
    motor_moved = pyqtSignal(int, float)  # motor_id, angle
    
    def __init__(self, ui_ref):
        super().__init__()
        self.ui = ui_ref
        
        # 추적 상태
        self.tracking_enabled = False
        self.current_target = None  # (x, y) 좌표
        
        # 모터 현재 위치 (각도)
        self.bottom_motor_angle = 0.0  # 좌우 회전
        self.top_motor_angle = 0.0     # 상하 회전
        
        # 카메라-LiDAR 매핑 파라미터
        self.camera_height = config.CAMERA_HEIGHT
        self.lidar_height = config.LIDAR_HEIGHT
        
        # 추적 파라미터
        self.update_threshold = config.TRACKING_ANGLE_THRESHOLD
        self.smooth_factor = config.TRACKING_SMOOTH_FACTOR
        
        # 주기적 업데이트 타이머
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_tracking)
        self.update_interval = config.TRACKING_UPDATE_INTERVAL
    
    def start_tracking(self):
        """추적 시작"""
        if not self.ui.state_icon.system_started:
            print("[TRACKING] 시스템이 시작되지 않음")
            return False
        
        if not self.ui.ser or not self.ui.ser.is_open:
            print("[TRACKING] 시리얼 포트 미연결")
            return False
        
        self.tracking_enabled = True
        self.update_timer.start(self.update_interval)
        self.tracking_started.emit()
        print("[TRACKING] 추적 시작")
        return True
    
    def stop_tracking(self):
        """추적 중지"""
        self.tracking_enabled = False
        self.update_timer.stop()
        self.current_target = None
        self.tracking_stopped.emit()
        print("[TRACKING] 추적 중지")
    
    def update_target(self, person_centroids):
        """
        LiDAR에서 감지된 사람 위치 업데이트
        
        Args:
            person_centroids: [(x1, y1), (x2, y2), ...] 사람들의 중심 좌표
        """
        if not self.tracking_enabled:
            return
        
        if not person_centroids:
            self.current_target = None
            return
        
        # 가장 가까운 사람 선택
        target = self._select_target(person_centroids)
        
        if target:
            self.current_target = target
            self.target_detected.emit(target[0], target[1])
    
    def _select_target(self, person_centroids):
        """
        추적할 대상 선택 (가장 가까운 사람)
        
        Args:
            person_centroids: 사람들의 중심 좌표 리스트
            
        Returns:
            (x, y) 선택된 타겟 좌표
        """
        if not person_centroids:
            return None
        
        # 거리 기준 정렬
        sorted_persons = sorted(
            person_centroids,
            key=lambda p: math.sqrt(p[0]**2 + p[1]**2)
        )
        
        return sorted_persons[0]  # 가장 가까운 사람
    
    def _update_tracking(self):
        """주기적으로 카메라 위치 업데이트"""
        if not self.current_target:
            return
        
        x, y = self.current_target
        
        # LiDAR 좌표를 카메라 각도로 변환
        bottom_angle, top_angle = self._lidar_to_camera_angles(x, y)
        
        # 현재 각도와 비교하여 이동 필요 여부 결정
        bottom_diff = abs(bottom_angle - self.bottom_motor_angle)
        top_diff = abs(top_angle - self.top_motor_angle)
        
        # 임계값 이상 차이나면 이동
        if bottom_diff > self.update_threshold:
            self._move_motor_smooth(1, bottom_angle)
        
        if top_diff > self.update_threshold:
            self._move_motor_smooth(2, top_angle)
    
    def _lidar_to_camera_angles(self, x, y):
        """
        LiDAR 좌표를 카메라 모터 각도로 변환
        
        Args:
            x: LiDAR X 좌표 (좌우, 미터)
            y: LiDAR Y 좌표 (전후, 미터)
            
        Returns:
            (bottom_angle, top_angle) 각도 (도)
        """
        # Bottom Motor (좌우 회전)
        # LiDAR 기준 각도 계산
        beta = math.degrees(math.atan2(x, y))
        
        # LiDAR 270도 회전 보정 적용
        bottom_angle = beta
        
        # Bottom Motor 각도 범위 제한 (-90 ~ 90)
        bottom_angle = max(config.BOTTOM_MOTOR_MIN, min(config.BOTTOM_MOTOR_MAX, bottom_angle))
        
        # Top Motor (상하 회전)
        # 거리 계산
        distance = math.sqrt(x**2 + y**2)
        
        # 높이 차이 고려
        height_diff = self.camera_height - self.lidar_height
        
        # 상하 각도 계산
        if distance > 0.1:  # 너무 가까우면 계산 안 함
            top_angle = math.degrees(math.atan2(height_diff, distance))
        else:
            top_angle = 0
        
        # Top Motor 각도 범위 제한 (-30 ~ 30)
        top_angle = max(config.TOP_MOTOR_MIN, min(config.TOP_MOTOR_MAX, top_angle))
        
        return bottom_angle, top_angle
    
    def _move_motor_smooth(self, motor_id, target_angle):
        """
        부드럽게 모터 이동
        
        Args:
            motor_id: 1=bottom, 2=top
            target_angle: 목표 각도
        """
        if motor_id == 1:
            current = self.bottom_motor_angle
        else:
            current = self.top_motor_angle
        
        # 부드러운 이동을 위한 중간 각도 계산
        smooth_angle = current + (target_angle - current) * self.smooth_factor
        
        # 정수 각도로 변환
        smooth_angle = round(smooth_angle)
        
        # 모터 이동 명령 전송
        cmd = f"s{motor_id}:{int(smooth_angle)}"
        
        if send_command(self.ui.ser, cmd):
            # 현재 위치 업데이트
            if motor_id == 1:
                self.bottom_motor_angle = smooth_angle
            else:
                self.top_motor_angle = smooth_angle
            
            self.motor_moved.emit(motor_id, smooth_angle)
            print(f"[TRACKING] Motor {motor_id} > {smooth_angle}°")
    
    def move_to_position(self, x, y):
        """
        특정 LiDAR 좌표로 즉시 이동
        
        Args:
            x, y: LiDAR 좌표
        """
        bottom_angle, top_angle = self._lidar_to_camera_angles(x, y)
        
        # Bottom Motor 이동
        cmd1 = f"s1:{int(bottom_angle)}"
        if send_command(self.ui.ser, cmd1):
            self.bottom_motor_angle = bottom_angle
            self.motor_moved.emit(1, bottom_angle)
        
        # Top Motor 이동
        cmd2 = f"s2:{int(top_angle)}"
        if send_command(self.ui.ser, cmd2):
            self.top_motor_angle = top_angle
            self.motor_moved.emit(2, top_angle)
        
        print(f"[TRACKING] Move to ({x:.2f}, {y:.2f}) > Bottom: {bottom_angle:.1f}°, Top: {top_angle:.1f}°")
    
    def calibrate_motors(self, bottom_angle=0, top_angle=0):
        """
        모터 현재 위치 보정
        
        Args:
            bottom_angle: Bottom Motor 현재 각도
            top_angle: Top Motor 현재 각도
        """
        self.bottom_motor_angle = bottom_angle
        self.top_motor_angle = top_angle
        print(f"[TRACKING] 보정 완료 - Bottom: {bottom_angle}°, Top: {top_angle}°")
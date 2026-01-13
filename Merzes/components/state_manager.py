"""
시스템 상태 관리 모듈
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from utils.resource_utils import resource_path


class StateManager:
    """시스템 상태 관리 클래스"""
    
    def __init__(self, ui_ref):
        self.ui = ui_ref
        
        # 초기 상태
        self.comport_connected = False
        self.camera_connected = False
        self.lidar_connected = False
        self.system_started = False
    
    def set_comport(self, is_connected):
        """COM 포트 연결 상태 설정"""
        self.comport_connected = is_connected
        self.update_state()

    def set_camera(self, is_connected):
        """카메라 연결 상태 설정"""
        self.camera_connected = is_connected
        self.update_state()

    def set_lidar(self, is_connected):
        """LiDAR 연결 상태 설정"""
        self.lidar_connected = is_connected
        self.update_state()

    def reset_all(self):
        """모든 상태 초기화"""
        self.comport_connected = False
        self.camera_connected = False
        self.lidar_connected = False
        self.update_state()

    def update_state(self):
        """상태에 따라 아이콘 업데이트"""
        # START 안 됐으면 무조건 RED
        if not self.system_started:
            self.show_red()
            return
        
        # START 됐고 모든 장치 연결됨
        if (self.system_started and 
            self.comport_connected and 
            self.camera_connected and 
            self.lidar_connected): 
            self.show_green()
        else:
            self.show_red()

    def show_red(self):
        """RED 상태 아이콘 표시"""
        image_path = resource_path("img/그룹 78.png")
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.state_label_circle.setPixmap(pixmap)

    def show_green(self):
        """GREEN 상태 아이콘 표시"""
        image_path = resource_path("img/그룹 79.png")
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.state_label_circle.setPixmap(pixmap)
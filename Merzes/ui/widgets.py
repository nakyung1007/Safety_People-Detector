from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QPainterPath, QRegion
from PyQt5.QtWidgets import QMenu, QPushButton


class RoundedMenu(QMenu):
    """둥근 모서리 메뉴"""
    
    def showEvent(self, event):
        super().showEvent(event)
        path = QPainterPath()
        rect = QRectF(0, 0, self.width(), self.height())
        radius = 15
        path.addRoundedRect(rect, radius, radius)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)


class ToggleButton(QPushButton):
    """토글 버튼 (클릭 릴리즈 시그널 추가)"""
    
    clicked_release = pyqtSignal()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # 버튼 영역 안에서 손을 뗀 경우에만 시그널 발생
        if self.rect().contains(event.pos()):
            self.clicked_release.emit()
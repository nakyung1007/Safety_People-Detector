import sys
import os
from PyQt5.QtWidgets import QApplication
import config

def main():
    # OpenCV FFMPEG 옵션 설정
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = config.OPENCV_FFMPEG_OPTIONS
    
    # Qt 애플리케이션 생성
    app = QApplication(sys.argv)
    
    from UI.main_window import MainWindow

    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    # 이벤트 루프 실행
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
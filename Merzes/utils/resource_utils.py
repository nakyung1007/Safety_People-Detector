import sys
import os


def resource_path(rel_path):
    """
    리소스 파일의 절대 경로를 반환합니다.
    PyInstaller로 빌드된 exe에서도 동작합니다.
    
    Args:
        rel_path: 상대 경로
        
    Returns:
        절대 경로
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # exe 내부 리소스
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, rel_path)
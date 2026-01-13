"""
LiDAR 데이터 처리 모듈
"""
import struct
import socket
import numpy as np
from sklearn.cluster import DBSCAN
import config


def hex_to_int32_signed(h: str) -> int:
    """16진수 문자열을 부호있는 32비트 정수로 변환"""
    v = int(h, 16)
    return v - (1 << 32) if v & (1 << 31) else v


def hex_to_float32(h: str) -> float:
    """16진수 문자열을 32비트 부동소수점으로 변환"""
    return struct.unpack(">f", bytes.fromhex(h))[0]


def parse_lmd(frame):
    """
    LMD 프레임을 파싱하여 각도와 거리 배열을 반환합니다.
    
    Args:
        frame: LMD 프레임 문자열
        
    Returns:
        tuple: (각도 배열, 거리 배열) 또는 (None, None)
    """
    if "DIST1" not in frame:
        return None, None

    toks = frame.split()
    try:
        i = toks.index("DIST1")
        scale = hex_to_float32(toks[i + 1])
        offset = hex_to_float32(toks[i + 2])
        start_deg = hex_to_int32_signed(toks[i + 3]) / 10000.0
        step_deg = int(toks[i + 4], 16) / 10000.0
        n_pts = int(toks[i + 5], 16)

        vals = toks[i + 6 : i + 6 + n_pts]
        dist = np.array([int(v, 16) * scale / 1000.0 + offset for v in vals])
        angles = np.deg2rad(start_deg + np.arange(n_pts) * step_deg)
        return angles, dist
    except:
        return None, None


def is_person(pts):
    """
    포인트 클러스터가 사람인지 판별합니다.
    
    Args:
        pts: numpy array of points (N x 2)
        
    Returns:
        bool: 사람 여부
    """
    n = len(pts)
    if not (config.PERSON_MIN_POINTS <= n <= config.PERSON_MAX_POINTS):
        return False

    width = np.ptp(pts[:, 0])
    height = np.ptp(pts[:, 1])
    if width > config.PERSON_MAX_WIDTH or height > config.PERSON_MAX_HEIGHT:
        return False

    ratio = height / (width + 1e-6)
    if not (config.PERSON_MIN_RATIO <= ratio <= config.PERSON_MAX_RATIO):
        return False

    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    if np.std(d) < config.PERSON_MIN_STD:
        return False

    return True


def cluster_and_classify(pts):
    """
    포인트를 클러스터링하고 사람을 분류합니다.
    
    Args:
        pts: numpy array of points (N x 2)
        
    Returns:
        tuple: (colors, person_centroids)
    """
    db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES).fit(pts)
    labels = db.labels_

    colors = []
    person_centroids = []
    
    for i, lab in enumerate(labels):
        if lab == -1:
            colors.append("gray")
        else:
            cluster = pts[labels == lab]
            if is_person(cluster):
                colors.append("red")
                if i == 0 or labels[i-1] != lab:  # 클러스터의 첫 포인트
                    cx, cy = cluster.mean(axis=0)
                    person_centroids.append((cx, cy))
            else:
                colors.append("blue")
    
    # x 기준 정렬
    person_centroids = sorted(person_centroids, key=lambda p: p[0])
    
    return colors, person_centroids


class LidarConnection:
    """LiDAR 소켓 연결 관리 클래스"""
    
    def __init__(self, host=None, port=None):
        self.host = host or config.LIDAR_HOST
        self.port = port or config.LIDAR_PORT
        self.sock = None
        self.connect()
    
    def connect(self):
        """LiDAR에 연결"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.sendall(config.STX + b"sEN LMDscandata 1" + config.ETX)
            self.sock.settimeout(config.LIDAR_SOCKET_TIMEOUT)
            print(f"[LIDAR] Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"[LIDAR] Connection failed: {e}")
            self.sock = None
    
    def reconnect(self):
        """재연결 시도"""
        print("[LIDAR] Reconnecting...")
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.connect()
    
    def receive(self):
        """데이터 수신"""
        if not self.sock:
            return None
        
        try:
            buf = self.sock.recv(65536)
            
            if config.STX not in buf or config.ETX not in buf:
                return None
            
            a, b = buf.find(config.STX), buf.find(config.ETX)
            msg = buf[a+1:b].decode(errors="ignore")
            
            if "LMDscandata" not in msg:
                return None
            
            return msg
        except:
            return None
    
    def close(self):
        """연결 종료"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
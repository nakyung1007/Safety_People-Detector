"""
카메라 스레드 모듈
"""
import os
import time
import cv2
import numpy as np
import datetime as dt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import config


class CameraThread(QThread):
    """카메라 스트리밍 및 녹화 스레드"""
    
    frame_signal = pyqtSignal(object)
    fps_signal = pyqtSignal(float)

    def __init__(self, ui_ref, rtsp_url, out_dir, fourcc="mp4v", target_fps=30):
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

        self.segment_sec = config.CAMERA_SEGMENT_SEC
        self.segment_start_time = None

    def get_today_folder(self):
        """날짜별 폴더 생성"""
        today = dt.datetime.now().strftime("%Y%m%d")
        folder = os.path.join(self.base_out_dir, today)

        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

    def create_writer(self):
        """VideoWriter 생성 (1분 단위 파일)"""
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

    def start_recording(self):
        """녹화 시작"""
        if self.recording:
            return

        self.writer = self.create_writer()
        if self.writer is None:
            return

        self.recording = True
        self.segment_start_time = time.time()
        print("[REC] 녹화 ON")

    def stop_recording(self):
        """녹화 중지"""
        self.recording = False
        print("[REC] 녹화 OFF")

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
        """메인 루프"""
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

        while self.running:
            try:
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
                    time.sleep(0.1)
                    continue

                self.ui.last_camera_frame_time = time.time()
                self.ui.state_icon.camera_connected = True

                if self.frame_size is None:
                    h, w = frame.shape[:2]
                    self.frame_size = (w, h)

                if self.recording:
                    if self.writer is None or not self.writer.isOpened():
                        print("[CameraThread] WARNING: writer invalid > recreate")
                        self.writer = self.create_writer()
                        if self.writer is None or not self.writer.isOpened():
                            print("[CameraThread] writer still invalid > skip frame")
                            continue

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

                    if time.time() - self.segment_start_time >= self.segment_sec:
                        try:
                            self.writer.release()
                        except:
                            pass
                        self.writer = self.create_writer()
                        self.segment_start_time = time.time()

                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    self.fps_signal.emit(fps)
                    start_time = time.time()
                    frame_count = 0

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
                print("[CameraThread] UNCAUGHT ERROR:", e)
                time.sleep(0.2)
                continue

        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            self.cap.release()

        if self.writer:
            try:
                self.writer.release()
            except:
                pass

        print("[CAM] Thread stopped")

    def stop(self):
        """스레드 종료"""
        self.running = False

        try:
            if hasattr(self, "cap") and self.cap.isOpened():
                self.cap.release()
                print("[CAM] cap.release()")
        except:
            pass

        try:
            if self.writer:
                self.writer.release()
                print("[CAM] writer.release()")
        except:
            pass

        self.wait()
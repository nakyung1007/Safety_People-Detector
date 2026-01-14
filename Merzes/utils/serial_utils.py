import time
import serial
import serial.tools.list_ports


def get_available_ports():
    """사용 가능한 시리얼 포트 목록을 반환합니다."""
    return [p.device for p in serial.tools.list_ports.comports()]


def connect_serial(port_name, baudrate=115200, timeout=0.1):
    """시리얼 포트에 연결합니다."""
    try:
        ser = serial.Serial(port_name, baudrate, timeout=timeout)
        time.sleep(0.1)
        print(f"[SERIAL] Connected > {port_name}")
        return ser
    except Exception as e:
        print(f"[ERROR] Connect failed: {e}")
        return None


def send_command(ser, cmd):
    """시리얼 명령을 전송합니다."""
    if ser is None or not ser.is_open:
        print("[ERROR] 포트 미연결")
        return False
    
    try:
        ser.write((cmd + "\n").encode())
        print(f"[SERIAL] {cmd}")
        return True
    except Exception as e:
        print(f"[ERROR] Send failed: {e}")
        return False


def check_servo_device(ser, timeout=1.0):
    """서보모터 장치 확인"""
    try:
        ser.reset_input_buffer()
        ser.write(b"ping\n")
        
        start = time.time()
        while time.time() - start < timeout:
            if ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                print("[SERVO CHECK]", line)
                
                if line == "SERVO_OK":
                    return True
        
        return False
    except Exception as e:
        print("[SERVO CHECK ERROR]", e)
        return False
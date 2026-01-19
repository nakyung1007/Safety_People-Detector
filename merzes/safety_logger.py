import os
import json
from datetime import datetime

class SafetyLogger:
    def __init__(self, base_path):
        self.log_dir = os.path.join(base_path, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    # 함수 이름을 'write_json_log'로 확실히 지정합니다.
    def write_json_log(self, data):
        today = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(self.log_dir, f"safety_log_{today}.json")
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"[LOGGER ERROR] {e}")
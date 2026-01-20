"""
YOLO 모델 정보 확인 (info.py)
모델 경로: ../checkpoints/yolo11m_safety.pt
"""

from ultralytics import YOLO
import torch
from pathlib import Path


def check_model_info():
    model_path = '../checkpoints/yolo11m_safety.pt'
    
    print("\n" + "=" * 70)
    print(f"🔍 YOLO 모델 정보 확인")
    print("=" * 70)
    print(f"\n모델 경로: {model_path}")
    
    # 파일 존재 확인
    if not Path(model_path).exists():
        print(f"\n❌ 모델 파일을 찾을 수 없습니다!")
        print(f"   경로: {Path(model_path).absolute()}")
        return
    
    try:
        # 모델 로드
        print("\n⏳ 모델 로딩 중...")
        model = YOLO(model_path)
        
        # 파일 크기
        file_size = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"\n💾 파일 크기: {file_size:.2f} MB")
        
        # 클래스 정보
        print(f"\n" + "=" * 70)
        print(f"🏷️  클래스 정보")
        print("=" * 70)
        print(f"\n   총 클래스 수: {len(model.names)}")
        print(f"\n   {'ID':<5} {'클래스명':<20} {'상태':<20}")
        print(f"   {'-'*5} {'-'*20} {'-'*20}")
        
        for idx, name in model.names.items():
            if name in ['no helmet', 'no vest']:
                status = "❌ 제거 예정"
            elif name in ['helmet', 'vest', 'person']:
                status = "✅ 유지"
            else:
                status = "❓ 확인 필요"
            
            print(f"   {idx:<5} {name:<20} {status:<20}")
        
        # 모델 구조
        print(f"\n" + "=" * 70)
        print(f"📊 모델 구조")
        print("=" * 70)
        
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"\n   전체 파라미터: {total_params:,}")
        print(f"   학습 가능 파라미터: {trainable_params:,}")
        
        # 체크포인트 정보
        print(f"\n" + "=" * 70)
        print(f"📅 학습 정보")
        print("=" * 70)
        print(f"\n   모델 타입: {model.task}")
        
        # 변경 계획
        print(f"\n" + "=" * 70)
        print(f"🔄 변경 계획")
        print("=" * 70)
        
        print(f"\n❌ 제거할 클래스:")
        for idx, name in model.names.items():
            if name in ['no helmet', 'no vest']:
                print(f"   - {idx}: {name}")
        
        print(f"\n✅ 유지할 클래스:")
        print(f"   - 0: helmet (유지)")
        print(f"   - 1: vest (유지)")
        print(f"   - 4 → 2: person (인덱스 변경)")
        
        print(f"\n⭐ 추가할 클래스:")
        new_classes = [
            (3, 'hook'),
            (4, 'forklift'),
            (5, 'crane'),
            (6, 'vehicle'),
            (7, 'yard_tractor')
        ]
        for idx, name in new_classes:
            print(f"   - {idx}: {name}")
        
        print(f"\n📊 통계:")
        print(f"   - 기존 클래스 수: {len(model.names)}")
        print(f"   - 새 클래스 수: 8")
        print(f"   - 제거: 2개")
        print(f"   - 유지: 3개")
        print(f"   - 추가: 5개")
        
        print("\n" + "=" * 70)
        print("✅ 모델 정보 확인 완료!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    check_model_info()
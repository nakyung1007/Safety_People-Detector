import os
import argparse
import wandb
from pathlib import Path
from ultralytics import YOLO

def setup_wandb(project_name="yolo11m-kfold-safety"):
    """WandB 로그인 및 설정 확인"""
    try:
        if not os.getenv('WANDB_API_KEY'):
            print("\n⚠️ WandB API Key 환경변수가 없습니다. 로그인이 필요할 수 있습니다.")
        return True
    except Exception as e:
        print(f"⚠️ WandB 설정 오류: {e}")
        return False

def train_fold(fold_idx, base_dir, args, group_name):
    """기존의 모든 학습 파라미터를 유지하며 개별 Fold 학습"""
    base_path = Path(base_dir)
    yaml_file = base_path / f'fold{fold_idx}.yaml'
    
    # 1. 실시간 차트를 위해 각 Fold마다 WandB 실행 시작
    run = wandb.init(
        project=args.wandb_project,
        group=group_name,
        name=f"fold_{fold_idx}",
        config={
            'fold': fold_idx,
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'optimizer': 'AdamW',
            'lr0': 0.001,
        },
        reinit=True
    )

    # 모델 로드
    model_path = '../checkpoints/yolo11m_safety.pt'
    model = YOLO(model_path) if Path(model_path).exists() else YOLO('yolo11n.pt')

    # 2. 학습 파라미터에 WandB 통합 추가
    results = model.train(
        data=str(yaml_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project='../checkpoints/kfold',
        name=f'fold{fold_idx}',
        
        # 나경 님의 기존 설정값들
        patience=50,
        save=True,
        save_period=10,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # 증강 설정 (Augmentation) 그대로 유지
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        
        device=0,
        workers=8,
        verbose=True,
        seed=0,
        deterministic=True,
        
        # ⭐ 핵심: Ultralytics의 WandB 통합 활성화
        plots=True,  # validation plot들도 wandb에 로깅
    )

    # 검증 및 결과 기록
    val_results = model.val(data=str(yaml_file))
    
    # 최종 검증 메트릭을 WandB에 요약으로 기록
    wandb.summary['final_map50'] = val_results.box.map50
    wandb.summary['final_map50_95'] = val_results.box.map
    wandb.summary['final_precision'] = val_results.box.mp
    wandb.summary['final_recall'] = val_results.box.mr
    
    # 개별 런 종료
    run.finish()
    
    return {
        'fold': fold_idx,
        'map50': val_results.box.map50,
        'map50_95': val_results.box.map,
        'precision': val_results.box.mp,
        'recall': val_results.box.mr
    }

def train_all_folds(args):
    # 고유한 그룹 이름 생성 (모든 폴드를 하나로 묶음)
    group_name = f"kfold_experiment_{wandb.util.generate_id()}"
    results = []

    for fold_idx in range(args.k):
        print(f"\n🚀 Fold {fold_idx} 학습 시작...")
        result = train_fold(fold_idx, args.base_dir, args, group_name)
        results.append(result)

    # 3. 전체 K-Fold 결과를 요약하는 별도 런 생성
    summary_run = wandb.init(
        project=args.wandb_project,
        group=group_name,
        name="kfold_summary",
        job_type="summary"
    )
    
    # 각 폴드별 최종 결과를 테이블로 기록
    table = wandb.Table(
        columns=["fold", "mAP50", "mAP50-95", "Precision", "Recall"],
        data=[[r['fold'], r['map50'], r['map50_95'], r['precision'], r['recall']] 
              for r in results]
    )
    wandb.log({"kfold_results_table": table})
    
    # 평균 메트릭 계산 및 기록
    avg_metrics = {
        'avg_map50': sum(r['map50'] for r in results) / len(results),
        'avg_map50_95': sum(r['map50_95'] for r in results) / len(results),
        'avg_precision': sum(r['precision'] for r in results) / len(results),
        'avg_recall': sum(r['recall'] for r in results) / len(results),
    }
    
    wandb.log(avg_metrics)
    
    # 요약 차트: 각 폴드별 성능 비교
    wandb.log({
        "fold_comparison": wandb.plot.bar(
            table, "fold", "mAP50",
            title="mAP50 by Fold"
        )
    })
    
    summary_run.finish()

    # 4. 파일 저장 (기존 기능 유지)
    results_file = Path(args.base_dir) / 'kfold_results.txt'
    with open(results_file, 'w') as f:
        f.write("K-Fold Cross Validation 결과 요약\n")
        f.write("="*50 + "\n\n")
        
        for r in results:
            f.write(f"Fold {r['fold']}:\n")
            f.write(f"  mAP50: {r['map50']:.4f}\n")
            f.write(f"  mAP50-95: {r['map50_95']:.4f}\n")
            f.write(f"  Precision: {r['precision']:.4f}\n")
            f.write(f"  Recall: {r['recall']:.4f}\n\n")
        
        f.write("="*50 + "\n")
        f.write("평균 메트릭:\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\n📄 결과 요약 저장 완료: {results_file}")
    print(f"📊 WandB 프로젝트: https://wandb.ai/your-username/{args.wandb_project}")

def main():
    parser = argparse.ArgumentParser(description='YOLO K-Fold with WandB Integration')
    parser.add_argument('--base_dir', type=str, default='../../Data/kfold_dataset')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--wandb_project', type=str, default='yolo11-kfold-safety')
    
    args = parser.parse_args()
    if setup_wandb(args.wandb_project):
        train_all_folds(args)

if __name__ == '__main__':
    main()
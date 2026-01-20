#!/bin/bash
# K-Fold Cross Validation 전체 학습 스크립트

echo '======================================'
echo 'Training Fold 0/4'
echo '======================================'
python retrain_yolo_kfold.py --fold 0
echo ''

echo '======================================'
echo 'Training Fold 1/4'
echo '======================================'
python retrain_yolo_kfold.py --fold 1
echo ''

echo '======================================'
echo 'Training Fold 2/4'
echo '======================================'
python retrain_yolo_kfold.py --fold 2
echo ''

echo '======================================'
echo 'Training Fold 3/4'
echo '======================================'
python retrain_yolo_kfold.py --fold 3
echo ''

echo '======================================'
echo 'Training Fold 4/4'
echo '======================================'
python retrain_yolo_kfold.py --fold 4
echo ''

echo 'All folds training completed!'

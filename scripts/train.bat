@echo off

set CONFIG="config/train.yaml"
python train.py --config %CONFIG%
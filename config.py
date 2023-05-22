import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = "dataset/"

cod_training_root = os.path.join(datasets_root, 'train/DUTS-TR')

pascal_path = os.path.join(datasets_root, 'test/img/PASCAL-S')
ecssd_path = os.path.join(datasets_root, 'test/img/ECSSD')
hku_path = os.path.join(datasets_root, 'test/img/HKU-IS')
dut_omron_path = os.path.join(datasets_root, 'test/img/DUT-OMRON')
dut_te_path = os.path.join(datasets_root, 'test/img/DUTS-TE')
sod_path = os.path.join(datasets_root, 'test/img/SOD')

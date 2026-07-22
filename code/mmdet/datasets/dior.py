# mmdet/datasets/dior.py
from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class DIORDataset(CocoDataset):
    """DIOR遥感图像数据集"""
    
    CLASSES = (
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 
        'bridge', 'chimney', 'dam', 'Expressway-Service-area', 
        'Expressway-toll-station', 'golffield', 'groundtrackfield', 
        'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 
        'tenniscourt', 'trainstation', 'vehicle', 'windmill'
    )
    
    def __init__(self, **kwargs):
        super(DIORDataset, self).__init__(**kwargs)
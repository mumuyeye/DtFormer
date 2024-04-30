
from .builder import DATASETS
from .cityscapes import CityscapesDataset

@DATASETS.register_module()
class FoggyCityscapesDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(FoggyCityscapesDataset, self).__init__(
            img_suffix='_leftImg8bit_foggy_beta_0.005.png',
            seg_map_suffix='_gtFine_labelTrainIds.png',
            **kwargs)
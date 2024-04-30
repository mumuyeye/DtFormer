# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .acdc import ACDCDataset, ACDCRefDataset, ACDCStyleDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .gta import GTADataset
from .synthia import SynthiaDataset
from .uda_dataset import UDADataset
from .uda_medium_dataset import UDAMediumDataset
from .foggy_zurich import FoggyZurichDataset, FoggyZurichLightDataset
from .foggy_driving import FoggyDrivingCoarseDataset, FoggyDrivingFineDataset
from .intermediate import IntermediateDataset
from .foggy_cityscapes import FoggyCityscapesDataset
from .uda_src_syn_imd_tgt_dataset import UDASynImdDataset
from .sat_image_related.cdl import CDLDataset, CDLMultiBandDataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'GTADataset',
    'SynthiaDataset',
    'UDADataset',
    'ACDCDataset',
    'DarkZurichDataset',
    'ACDCRefDataset',
    'UDAMediumDataset',
    'FoggyZurichDataset',
    'FoggyDrivingCoarseDataset',
    'FoggyDrivingFineDataset',
    'IntermediateDataset',
    'FoggyZurichLightDataset',
    'FoggyCityscapesDataset',
    'UDASynImdDataset',
    'CDLDataset',
    'CDLMultiBandDataset',
    'ACDCStyleDataset'
]

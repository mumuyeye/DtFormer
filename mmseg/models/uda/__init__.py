# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.compare.dacs import DACS
from mmseg.models.uda.compare.NoUDA import NoUDA
from mmseg.models.uda.CuDAFormer import CuDAFormer
from mmseg.models.uda.CuDAFormer_milestone import CuDAFormerMilestone

from mmseg.models.uda.daformer import DAFormer
from mmseg.models.uda.compare.adaptsegnet import AdaptSegNet
from mmseg.models.uda.compare.advent import ADVENT
from mmseg.models.uda.SAM_enhanced_DAFormer import DAFormerEnhancedBySAM
from mmseg.models.uda.CuDAFormer_enhanced_by_SAM import CuDAFormerEnhancedBySAM
from mmseg.models.uda.hrda import HRDA
from mmseg.models.uda.mic import MIC


__all__ = ['DACS', 'NoUDA', 'CuDAFormer', 'CuDAFormerMilestone','DAFormerEnhancedBySAM',
           'CuDAFormerEnhancedBySAM',
           # --------compare--------
           'AdaptSegNet', 'ADVENT', 'DAFormer', 'HRDA', 'MIC']

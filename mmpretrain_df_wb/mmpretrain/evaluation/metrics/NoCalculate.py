# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import tempfile
from typing import List, Optional

from mmengine.evaluator import BaseMetric
from mmengine.utils import track_iter_progress

from mmpretrain.registry import METRICS
from mmpretrain.utils import require


@METRICS.register_module()
class NoCalculate(BaseMetric):
    """
    啥也不算，用于byol等模型
    """

    @require('pycocoevalcap')
    def __init__(self,):
        super().__init__()

    def process(self, data_batch, data_samples):
        a = 1

    def compute_metrics(self, results: List):
        print('results:', results)


from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SemiTransform(BaseTransform):
    def __init__(self, split_path):
        super().__init__()
        self.split_path = split_path

    def transform(self, results: dict):
        results = results
        # 筛选数据
        # with open(self.split_path, mode="r") as file:


        return results



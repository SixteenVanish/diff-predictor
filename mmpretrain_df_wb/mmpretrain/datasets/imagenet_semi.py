# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .categories import IMAGENET_CATEGORIES
from .custom import CustomDataset


@DATASETS.register_module()
class ImageNet_Semi(CustomDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': IMAGENET_CATEGORIES}

    def __init__(self,
                 split_path,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            if split == 'test':
                logger = MMLogger.get_current_instance()
                logger.info(
                    'Since the ImageNet1k test set does not provide label'
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

        self.split_path = split_path

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

    def load_data_list(self):
        data_list = super().load_data_list()

        labeled_data = []
        with open(self.split_path, 'r') as file:
            labeled_data = file.readlines()

        labeled_data_list = []
        unlabeled_data_list = []
        for img in data_list:
            img_path = img['img_path']
            img_name = img_path.split('/')[-1]
            if img_name in labeled_data:
                labeled_data_list.append(img)
            else:
                unlabeled_data_list.append(img)

        return labeled_data_list, unlabeled_data_list



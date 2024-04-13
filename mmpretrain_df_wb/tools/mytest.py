# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
import pdb
import itertools
from PIL import Image
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import rich.progress as progress
import torch
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.device import get_device
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmpretrain.apis import get_model
from mmpretrain.registry import DATASETS
from mmpretrain.visualization import UniversalVisualizer
from tools.visualization.browse_dataset import make_grid

# from tools.visualization.browse_dataset import make_grid, InspectCompose
from mmengine.dataset import Compose

try:
    from sklearn.manifold import TSNE
except ImportError as e:
    raise ImportError('Please install `sklearn` to calculate '
                      'TSNE by `pip install scikit-learn`') from e
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--test-cfg',
        help='test config file path to load config of test dataloader.')
    parser.add_argument(
        '--extract-stage',
        choices=['backbone', 'neck', 'DM', 'pre_logits', 'target_backbone', 'target_neck', 'predictor_mid', 'predictor'],
        help='The visualization stage of the model')
    parser.add_argument(
        '--class-idx',
        nargs='+',
        type=int,
        help='The categories used.')
    parser.add_argument(
        '--max-num-class',
        type=int,
        default=200,
        help='The first N categories to test. '
        'Defaults to 200.')
    parser.add_argument(
        '--max-num-samples',
        type=int,
        default=50,
        help='The maximum number of samples per category. '
        'Higher number need longer time to calculate. Defaults to 50.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--device')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--feature-vis', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    test_work_dir = osp.join(cfg.work_dir, f'test_{timestamp}/')
    mkdir_or_exist(osp.abspath(test_work_dir))

    # init the logger before other steps
    log_file = osp.join(test_work_dir, 'test.log')
    logger = MMLogger.get_instance(
        'mmpretrain',
        logger_name='mmpretrain',
        log_file=log_file,
        log_level=cfg.log_level)

    # build the model from a config file and a checkpoint file
    device = args.device or get_device()
    model = get_model(cfg, args.checkpoint, device=device)
    logger.info('Model loaded.')

    # build the dataset
    if args.test_cfg is not None:
        dataloader_cfg = Config.fromfile(args.test_cfg).get('test_dataloader')
    elif 'test_dataloader' not in cfg:
        raise ValueError('No `test_dataloader` in the config, you can '
                         'specify another config file that includes test '
                         'dataloader settings by the `--test-cfg` option.')
    else:
        dataloader_cfg = cfg.get('test_dataloader')

    dataset = DATASETS.build(dataloader_cfg.pop('dataset'))
    classes = dataset.metainfo.get('classes')
    
    if args.class_idx is None:
        num_classes = args.max_num_class if classes is None else len(classes)
        args.class_idx = list(range(num_classes))[:args.max_num_class]

    if classes is not None:
        classes = [classes[idx] for idx in args.class_idx]
    else:
        classes = args.class_idx

    # compress dataset, select that the label is less then max_num_class
    subset_idx_list = []
    counter = defaultdict(int)
    for i in range(len(dataset)):
        gt_label = dataset.get_data_info(i)['gt_label']
        if (gt_label in args.class_idx
                and counter[gt_label] < args.max_num_samples):
            subset_idx_list.append(i)
            counter[gt_label] += 1
    dataset.get_subset_(subset_idx_list)
    logger.info(f'test {len(subset_idx_list)} samples.')
    
    dataloader_cfg.dataset = dataset
    dataloader_cfg.setdefault('collate_fn', dict(type='default_collate'))
    dataloader = Runner.build_dataloader(dataloader_cfg)
    
    if args.decode:
        cfg.visualizer.pop('type')
        fig_cfg = dict(figsize=(16, 10))
        # pdb.set_trace()
        visualizer = UniversalVisualizer(
            **cfg.visualizer, fig_show_cfg=fig_cfg, fig_save_cfg=fig_cfg)
        visualizer.dataset_meta = dataset.metainfo
    if args.feature_vis:
        labels_list = []
        features_list = []
        
    count = 0
    label_centers = []
    label_features = []
    for data in progress.track(dataloader, description='test...'):
        with torch.no_grad():     
            
            a = data['inputs'][0][0]
            data_input = np.transpose(a.cpu().numpy(), (1, 2, 0))   # [64,64,3] 相当于intermediate_imgs[0]['img']
            data_sample = data['data_samples'][0]
            visualizer.visualize_cls(
                data_input[..., ::-1],
                data_sample,
                name="zeros.png",
                out_file=test_work_dir + "zeros.png")

            # preprocess data
            data = model.data_preprocessor(data)    # <class 'dict'>
            # data['inputs']为list，list中为不同view的shape为[bsz, 3, img_size, img_size]的tensor
            # data['data_samples']为list，list中为bsz个<class 'mmpretrain.structures.data_sample.DataSample'>
            batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
            
            # extract features
            extract_args = {}
            if args.extract_stage:
                extract_args['stage'] = args.extract_stage
            batch_features = model.extract_feat(batch_inputs, **extract_args)[0]
            
            if args.decode:
                # decode
                batch_images_decoded = model.decoder(batch_features)
                
                # visualization
                bsz = batch_images_decoded.shape[0]
                bsz = 3
                if False:
                    # print(batch_inputs[0][-1])
                    # print(batch_images_online[-1])
                    # print(F.mse_loss(batch_inputs[0].float(), batch_images_online.float(), reduction="mean"))
                    for idx in range(0, bsz):
                        data_sample = batch_data_samples[idx]
                        gt_label = int(data_sample.gt_label)
                        img_path = data_sample.img_path
                        file_name = img_path[-16:-5]
                        
                        # print(batch_images_online[idx])
                        # print(reverse_norm(batch_images_online[idx][0]))
                        # print(reverse_norm(batch_images_online[idx][1]))
                        # print(reverse_norm(batch_images_online[idx][2]))
                        data_input = np.transpose(reverse_norm(batch_inputs[0][idx]).cpu().numpy(), (1, 2, 0))
                        pic_name_t = f"transformed_{file_name}_label{gt_label}_{idx}.png"
                        visualizer.visualize_cls(
                            data_input,
                            data_sample,
                            name=pic_name_t,
                            out_file=test_work_dir + pic_name_t)
                        
                        image_online = np.transpose(reverse_norm(batch_images_online[idx]).cpu().numpy(), (1, 2, 0))
                        pic_name_d = f"decoded_{file_name}_label{gt_label}_{idx}.png"
                        visualizer.visualize_cls(
                            image_online,
                            data_sample,
                            name=pic_name_d,
                            out_file=test_work_dir + pic_name_d,
                            )
                        exit(0)
                if True:   # 使用 make_grid
                    for idx in range(0, bsz):
                        data_sample = batch_data_samples[idx]
                        gt_label = int(data_sample.gt_label)
                        img_path = data_sample.img_path
                        # file_name = img_path[-16:-5]
                        
                        print(idx)
                        print('before')
                        print('mean data_input', np.mean(batch_inputs[0][idx].cpu().numpy()))
                        print('mean image_decoded', np.mean(batch_images_decoded[idx].cpu().numpy()))
                        print('std data_input', np.std(batch_inputs[0][idx].cpu().numpy()))
                        print('std image_decoded', np.std(batch_images_decoded[idx].cpu().numpy()))
                        
                        data_input = np.transpose(reverse_norm(batch_inputs[0][idx]).cpu().numpy(), (1, 2, 0))
                        image_decoded = np.transpose(reverse_norm(batch_images_decoded[idx]).cpu().numpy(), (1, 2, 0))
                        
                        print('after')
                        print('mean data_input', np.mean(data_input))
                        print('mean image_decoded', np.mean(image_decoded))
                        print('std data_input', np.std(data_input))
                        print('std image_decoded', np.std(image_decoded))
                        
                        images = make_grid([data_input.astype('uint8'), image_decoded.astype('uint8')], ['transformed', 'decoded'])
                        pic_name = f"label{gt_label}_{idx}.png"
                        # pic_name = f"{file_name}_label{gt_label}_{idx}.png"
                        visualizer.visualize_cls(
                            images,
                            data_sample,
                            name=pic_name,
                            out_file=test_work_dir + pic_name,
                            )
                        # exit(0)
      
            if args.analysis:
                count += 1
                if count <= 10:
                    distributionAnalysis(features_tuple = [batch_features], root_path = test_work_dir, gt_label = gt_label)
                label_centers.append(torch.mean(batch_features, dim=0))
                label_features.append(batch_features)
            
            if args.feature_vis:
                batch_labels = [int(data_sample.gt_label) for data_sample in batch_data_samples]
                labels_list += batch_labels
                features_list += batch_features     
      
    if args.analysis:
        print("\ninter-class")
        
        # 每个类计算一个center，然后计算cos sim
        labelCenters = torch.stack(label_centers)
        num_class, inter_cos_sim = cal_cos_sim(labelCenters)
        print('average label centers cosine similarity: ', torch.sum(inter_cos_sim) / (num_class * num_class - num_class))
        
        all_features = torch.cat(label_features, dim=0)
        # 计算所有特征之间的余弦相似度
        dot_product = torch.matmul(all_features, all_features.t())
        norms = torch.norm(all_features, dim=1, keepdim=True)
        similarities = dot_product / (norms * norms.t() + 1e-8)  # 避免除以0
        # 排除同类之间的相似度
        num_classes = len(label_features)
        mask = torch.ones_like(similarities)
        for i in range(num_classes):
            start_idx = sum(len(features) for features in label_features[:i])
            end_idx = start_idx + len(label_features[i])
            mask[start_idx:end_idx, start_idx:end_idx] = 0
        # 计算平均相似度
        total_similarity = torch.sum(similarities * mask)
        num_pairs = torch.sum(mask).item()
        average_similarity = total_similarity / num_pairs
        print('average inter-class samples cosine similarity: ', average_similarity.item())
    
    if args.feature_vis:
        
        classess_list = ['goldfish', 'tabby cat', 'Persian cat', 'Egyptian cat', 'puma', 'lion', 'maltese', 'japanese spaniel',
                       'English foxhound', 'balloon']
        
        # tsne
        fit_model = TSNE(
            n_components=2,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate=200.0,
            n_iter=1000,
            n_iter_without_progress=300,
            init='random')
        logger.info('Running t-SNE.')
        features_tensor = torch.stack(features_list, dim=0)
        result = fit_model.fit_transform(features_tensor.cpu())
        res_min, res_max = result.min(0), result.max(0)
        res_norm = (result - res_min) / (res_max - res_min)

        _, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            res_norm[:, 0],
            res_norm[:, 1],
            alpha=1.0,
            s=15,
            c=labels_list,
            cmap='tab20')
        legend = ax.legend(scatter.legend_elements()[0], classess_list)
        ax.add_artist(legend)
        plt.savefig(f'{test_work_dir}tsne_vis.png')
        
        # original
        _, ax = plt.subplots(figsize=(10, 10))
        features_tensor = features_tensor.cpu()
        scatter = ax.scatter(
            features_tensor[:, 0],
            features_tensor[:, 1],
            alpha=1.0,
            s=7,
            c=labels_list,
            cmap='tab20')
        legend = ax.legend(scatter.legend_elements()[0], classess_list)
        ax.add_artist(legend)
        plt.savefig(f'{test_work_dir}original_vis.png')
        
        
        logger.info(f'Save features and results to {test_work_dir}')
             
    
def distributionAnalysis(
        features_tuple,
        root_path,
        gt_label
    ):
        pic_name = f'label{gt_label}_cos_sim_Histogram.png'
        count = 0
        
        for features in features_tuple:  # [bsz, dim]
            count += 1
            bsz, cos_sim = cal_cos_sim(features)
            variance = torch.var(features, dim=0)
            std_deviation = torch.std(features, dim=0)
            coefficient_of_variation = (torch.std(features, dim=0) / torch.mean(features, dim=0))   # 离散系数
            covariance_matrix = torch.mm(features.t(), features) / bsz

            print(f"label {gt_label}")
            print('average cosine similarity: ', torch.sum(cos_sim) / (bsz * bsz - bsz))
            # print('average variance: ', torch.mean(variance))
            # print('average std_deviation: ', torch.mean(std_deviation))
            # print('coefficient of variation: ', coefficient_of_variation)
            
            # 绘制直方图
            bsz = cos_sim.size(0)
            cos_sim_flat = cos_sim.view(-1)
            cos_sim_flat = cos_sim_flat[cos_sim_flat != 0]  # 去除对角线上的相似度
            plt.hist(cos_sim_flat.cpu().numpy(), bins=100)
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.title('Histogram of Cosine Similarity')
            # plt.show()
            path = osp.join(root_path, pic_name)
            plt.savefig(path)  # 保存图片
            plt.close()
     
def cal_cos_sim(features):
    bsz = features.shape[0]
    norms = torch.norm(features, dim=1, keepdim=True)
    normalized_features = features / norms
    cos_sim = torch.matmul(normalized_features, normalized_features.t())  # [bsz, bsz]
    cos_sim = cos_sim * (1 - torch.eye(bsz)).to(features.device)    # 对角线处理，令其皆为0
    return bsz, cos_sim
        
def reverse_norm(
    data,   # [3, img_size, img_size] or [bsz, 3, img_size, img_size]
    mean=[255*0.4802, 255*0.4481, 255*0.3975],
    std=[255*0.2302, 255*0.2265, 255*0.2262]
    ):
    mean_tensor = torch.tensor(mean).view(3, 1, 1).to(data.device)
    std_tensor = torch.tensor(std).view(3, 1, 1).to(data.device)
    data = data * std_tensor + mean_tensor
    data = data.clamp_(max=255-(1e-8))
    return data
                
def save_one_pic(
    sample_data, path, rgb=True, reverseNorm=True,
    mean=[255*0.4802, 255*0.4481, 255*0.3975],
    std=[255*0.2302, 255*0.2265, 255*0.2262],
    ): 
    if reverseNorm:
        sample_data = reverse_norm(sample_data, mean, std)
    if rgb:
        sample_data = sample_data[[2,0,1], :, :]   # RGB <-> BGR
    image_data = np.transpose(sample_data.cpu().numpy(), (1, 2, 0))     # 调整通道位置
    # image_data *= 255
    # image_data = 255 - image_data
    
    image = Image.fromarray((image_data).astype(np.uint8))    # 使用PIL创建图像对象
    image.save(path)


if __name__ == '__main__':
    main()

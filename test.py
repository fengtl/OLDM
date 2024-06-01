# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from parser_train import parser_, relative_path_to_absolute_path
from tqdm import tqdm
import mmcv
import cv2
from PIL import Image, ImageFont, ImageDraw

from data import create_dataset
from models import adaptation_modelv2
from metrics import runningScore


import tsne

def test(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger)
    if opt.open:
        print('\ntest in open dataset:')
        logger.info('test in open dataset:')
    else:
        print('\ntest in compound dataset:')
        logger.info('test in compound dataset:')
 
    if opt.model_name == 'deeplabv2':
        checkpoint = torch.load(opt.resume_path)
        model = adaptation_modelv2.CustomModel(opt, logger)

        # # for DACS ckpt
        # dict_tmp = {}
        # for k, v in checkpoint['model'].items():
        #     dict_tmp['module.'+k] = v
        # model.BaseNet_DP.load_state_dict(dict_tmp, strict=True)

        dict_tmp = {}
        for k, v in checkpoint['ResNet101']['model_state'].items():
            dict_tmp['module.'+k] = v
        model.BaseNet_DP.load_state_dict(dict_tmp, strict=True)

        # tmp
        # model.BaseNet_DP.load_state_dict(checkpoint['ResNet101']['model_state'], strict=True)

        try:
            model.AuxDecoder.load_state_dict(checkpoint['Classifier_Module2']['model_state'], strict=True)
        except:
            dict_tmp = {}
            for k, v in checkpoint['ResNet101']['model_state'].items():
                if k.startswith('layer5'):
                    dict_tmp[k[7:]] = v
            model.AuxDecoder.load_state_dict(dict_tmp, strict=True)

    running_metrics_val = runningScore(opt.n_class)

    if opt.use_mem_test:
        try:
            model.mem = checkpoint['ResNet101']['mem']
        except:
            model.mem = torch.load(opt.mem_resume_path)
        print('\ntest with memory:')
        logger.info('test with memory:')
        validation(model, logger, datasets, device, running_metrics_val, opt=opt, use_mem=True)
    else:
        print('\ntest without memory:')
        logger.info('test without memory:')
        validation(model, logger, datasets, device, running_metrics_val, opt=opt, use_mem=False)

def validation(model, logger, datasets, device, running_metrics_val, opt, use_mem):
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(datasets.target_valid_loader, device, model, running_metrics_val, use_mem, opt.use_mem_select_num)

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))
    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))
    running_metrics_val.reset()

    torch.cuda.empty_cache()

def validate(valid_loader, device, model, running_metrics_val, use_mem, select_num):
    sm = torch.nn.Softmax(dim=1)
    i = 0
    for data_i in tqdm(valid_loader):
        i = i + 1
        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)

        if use_mem:
            outs = model.BaseNet_DP(images_val, mem=model.mem, select_num=select_num, aux_decoder=model.AuxDecoder_DP)
        else:
            outs = model.BaseNet_DP(images_val, mem=None, select_num=select_num, aux_decoder=model.AuxDecoder_DP)

        # outputs = F.interpolate(sm(outs['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
        outputs = F.interpolate(outs['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

        # # for visual origin pred gt
        # dir_name = os.path.join('work_dirs', 'test', 'visual_tmp',
        #                         os.path.basename(os.path.dirname(data_i['img_path'][0])),
        #                         os.path.basename(data_i['img_path'][0]))
        # outputs_sm = F.interpolate(sm(outs['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True).cpu()
        # assert outputs_sm.shape[0] == 1
        # palette = np.array(
        #     [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
        #      (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        #      (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)],
        #     dtype=np.uint8)
        # color_pred = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
        # for label, color in enumerate(palette):
        #     color_pred[pred.squeeze(0) == label, :] = color
        # color_pred = color_pred[..., ::-1]  # convert to BGR
        # color_gt = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        # for label, color in enumerate(palette):
        #     color_gt[gt.squeeze(0) == label, :] = color
        # color_gt = color_gt[..., ::-1]  # convert to BGR
        # mmcv.imwrite(np.array(Image.open(data_i['img_path'][0]).resize((gt.shape[2], gt.shape[1]))), dir_name.replace('.png', '_origin.png'))
        # mmcv.imwrite(color_gt, dir_name.replace('.png', '_gt.png'))
        # mmcv.imwrite(color_pred, dir_name.replace('.png', '_pred.png'))

        # # for error pixel
        # dir_name = os.path.join('work_dirs', 'visualization', 'xy_44',
        #                         os.path.basename(os.path.dirname(data_i['img_path'][0])),
        #                         os.path.basename(data_i['img_path'][0]))
        # mask1 = torch.where(
        #     torch.tensor(pred) != torch.tensor(gt),
        #     torch.ones(pred.shape),
        #     torch.zeros(pred.shape))
        # mask2 = torch.where(
        #     torch.tensor(gt) != torch.full(pred.shape, 250),
        #     torch.ones(pred.shape),
        #     torch.zeros(pred.shape))
        # mask = torch.mul(mask1, mask2).squeeze(0)
        # error_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # error_map[mask == 1, :] = [0, 0, 255]
        # mmcv.imwrite(error_map, dir_name.replace('.jpg', '_error_pseudo.jpg'))

        # # for uncertainty
        # segment_maps = list()
        # for select_num in [1, 5, 10, 15, 20]:
        #     outs = model.BaseNet_DP(images_val, mem=mem, select_num=select_num)
        #     outputs_sm = F.interpolate(sm(outs['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
        #     segment_maps.append(outputs_sm)
        # segment_maps = torch.cat(segment_maps)
        # uncertainty_map = torch.sum(
        #     torch.sub(torch.max(segment_maps, dim=0).values, torch.min(segment_maps, dim=0).values),
        #     dim=0, keepdim=True)  # 1*H*W
        # uncertainty_map = 1 - torch.div(uncertainty_map-uncertainty_map.min(), uncertainty_map.max()-uncertainty_map.min())
        # b, g, r = cv2.split(cv2.applyColorMap(np.uint8(uncertainty_map.cpu().squeeze(0)*255), cv2.COLORMAP_JET))
        # uncertainty_map = cv2.merge([r, g, b])
        # dir_name = os.path.join('work_dirs', 'visualization', 'xy_44',
        #                         os.path.basename(os.path.dirname(data_i['img_path'][0])),
        #                         os.path.basename(data_i['img_path'][0]))
        # mmcv.imwrite(uncertainty_map, dir_name.replace('.jpg', '_uncertainty.jpg'))

        # # for visualization
        # dir_name = os.path.join('work_dirs', 'visualization', 'xy_44',
        #                         os.path.basename(os.path.dirname(data_i['img_path'][0])),
        #                         os.path.basename(data_i['img_path'][0]))
        # outputs_sm = F.interpolate(sm(outs['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True).cpu()
        # assert outputs_sm.shape[0] == 1
        # conf, _ = outputs_sm[0].max(0, keepdim=True)
        # b, g, r = cv2.split(cv2.applyColorMap(np.uint8(conf.squeeze(0)*255), cv2.COLORMAP_JET))
        # conf_hot_map = cv2.merge([r, g, b])
        # palette = np.array(
        #     [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
        #      (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        #      (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)],
        #     dtype=np.uint8)
        # color_pred = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
        # for label, color in enumerate(palette):
        #     color_pred[pred.squeeze(0) == label, :] = color
        # color_pred = color_pred[..., ::-1]  # convert to BGR
        # color_gt = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        # for label, color in enumerate(palette):
        #     color_gt[gt.squeeze(0) == label, :] = color
        # color_gt = color_gt[..., ::-1]  # convert to BGR
        # mmcv.imwrite(mmcv.imread(data_i['img_path'][0]), dir_name.replace('.jpg', '_origin.jpg'))
        # mmcv.imwrite(color_gt, dir_name.replace('.jpg', '_gt.jpg'))
        # mmcv.imwrite(color_pred, dir_name.replace('.jpg', '_pred_wo.jpg'))
        # np.save(dir_name.replace('.jpg', '_conf_wo.npy'), conf)
        # mmcv.imwrite(conf_hot_map, dir_name.replace('.jpg', '_conf_hot_wo.jpg'))

        # # for draw tsne
        # dir_name = os.path.join(os.path.dirname(opt.resume_path), 'tsne', 'trg_val')
        # if not os.path.exists(os.path.dirname(dir_name)):
        #     os.mkdir(os.path.dirname(dir_name))
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        # torch.save(outs['pre_compensate_feat'], os.path.join(dir_name, 'pre_{}'.format(i)))
        # torch.save(outs['feat'], os.path.join(dir_name, 'post_{}'.format(i)))
        # torch.save(gt, os.path.join(dir_name, 'gt_{}'.format(i)))
        # torch.save(outs['out'], os.path.join(dir_name, 'predict_{}'.format(i)))

def get_logger(opt):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(opt.logdir, os.path.basename(opt.resume_path).replace('.pth', '_test_{}.log'.format('mem' if opt.use_mem_test else 'no_mem')))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = os.path.dirname(opt.resume_path)

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt)

    test(opt, logger)

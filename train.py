import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from zipfile import ZipFile
from tqdm import tqdm

from data import create_dataset
from utils import get_logger
from models import adaptation_modelv2
from metrics import runningScore, averageMeter
from parser_train import parser_, relative_path_to_absolute_path

def train(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger)

    if opt.model_name == 'deeplabv2':
        model = adaptation_modelv2.CustomModel(opt, logger)

    # Setup Metrics
    running_metrics_val_w_mem = runningScore(opt.n_class)
    running_metrics_val_wo_mem = runningScore(opt.n_class)
    time_meter = averageMeter()

    # load category anchors
    if opt.stage=='mem' or opt.stage=='denoise':
        objective_vectors = torch.load(os.path.join(
            os.path.dirname(opt.resume_path), 'prototypes'))
        model.objective_vectors = torch.Tensor(objective_vectors).to(0)

    # # save path of memory
    # if opt.stage == 'mem' or opt.stage=='denoise':
    #     mem_save_path = os.path.join(opt.logdir, 'memory')
    #     if not os.path.exists(mem_save_path):
    #         os.mkdir(mem_save_path)
    
    # begin training
    model.iter = 0
    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        for data_i in datasets.target_train_loader:
            target_image = data_i['img'].to(device)
            target_imageS = data_i['img_strong'].to(device)
            target_params = data_i['params']
            target_image_full = data_i['img_full'].to(device)
            target_weak_params = data_i['weak_params']

            target_lp = data_i['lp'].to(device) if 'lp' in data_i.keys() else None
            target_lpsoft = data_i['lpsoft'].to(device) if 'lpsoft' in data_i.keys() else None

            source_data = datasets.source_train_loader.next()
            
            model.iter += 1
            i = model.iter
            images = source_data['img'].to(device)
            labels = source_data['label'].to(device)
            source_imageS = source_data['img_strong'].to(device)
            source_params = source_data['params']

            start_ts = time.time()

            model.train(logger=logger)
            if opt.freeze_bn:
                model.freeze_bn_apply()

            model.optimizer_zerograd()

            if opt.stage == 'warm_up':
                loss_src, loss_adv_G, loss_D = model.step_adv(
                    images, labels, target_image, source_imageS, source_params)
            elif opt.stage == 'mem':
                loss_src, loss_trg, loss_adv_G, loss_D, loss_align, loss_aux_decoder = model.step_mem(
                    images, labels, target_image, target_lpsoft)
            elif opt.stage == 'denoise':
                loss, loss_CTS, loss_consist,mem = model.step_denoise_pixel(
                    images, labels, target_image, target_imageS, target_params,
                    target_lp, target_lpsoft, target_image_full, target_weak_params)
                # loss, loss_CTS, loss_consist = model.step_denoise(
                #     images, labels, target_image, target_imageS, target_params,
                #     target_lp, target_lpsoft, target_image_full, target_weak_params)
            elif opt.stage=='distillation' or opt.stage=='distillation_pro':
                loss_src, loss_trg, loss_kld_out, loss_kld_feat, mem = model.step_distillation_update_mem(
                    images, labels, target_image, target_imageS, target_params, target_lp)
            else:
                print('Error: incorrect stage!')

            time_meter.update(time.time() - start_ts)

            #print(i)
            if (i + 1) % opt.print_interval == 0:
                if opt.stage == 'warm_up':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_src: {:.4f}  loss_adv_G: {:.4f}  loss_D: {:.4f} loss_total: {:.4f} Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        epoch+1, opt.epochs, i + 1, opt.train_iters, loss_src, loss_adv_G, loss_D, loss_src+loss_adv_G+loss_D, time_meter.avg / opt.bs)
                elif opt.stage == 'mem':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_src: {:.4f}  loss_trg: {:.4f}  loss_adv_G: {:.4f}  loss_D: {:.4f} loss_align: {:.4f} loss_aux_decoder: {:.4f} loss_total: {:.4f} Time/Image: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_src, loss_trg, loss_adv_G, loss_D, loss_align, loss_aux_decoder, loss_src+loss_trg+loss_adv_G+loss_D+loss_align+loss_aux_decoder, time_meter.avg / opt.bs)
                elif opt.stage == 'denoise':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss: {:.4f}  loss_CTS: {:.4f}  loss_consist: {:.4f} Time/Image: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss, loss_CTS, loss_consist, time_meter.avg / opt.bs)
                else:
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_src: {:.4f}  loss_trg: {:.4f}  loss_kld_out: {:.4f}  loss_kld_feat: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_src, loss_trg, loss_kld_out, loss_kld_feat, time_meter.avg / opt.bs)
                print(print_str)
                logger.info(print_str)
                time_meter.reset()

            # evaluation
            if (i + 1) % opt.val_interval == 0:
                validation(model, logger, datasets, device, running_metrics_val_w_mem, running_metrics_val_wo_mem, model.iter, opt)
                torch.cuda.empty_cache()
                logger.info('Best iou with mem until now is {}'.format(model.best_iou_w_mem))
                logger.info('Best iou without mem until now is {}'.format(model.best_iou_wo_mem))

            model.scheduler_step()

def validation(model, logger, datasets, device, running_metrics_val_w_mem, running_metrics_val_wo_mem, iters, opt):
    iters = iters
    _k = -1
    for v in model.optimizers:
        _k += 1
        for param_group in v.param_groups:
            _learning_rate = param_group.get('lr')
        logger.info("learning rate is {} for {} net".format(_learning_rate, model.nets[_k].__class__.__name__))

    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        if opt.stage != 'warm_up':
            print('test with memory:')
            logger.info('test with memory:')
            validate(datasets.target_valid_loader, device, model, running_metrics_val_w_mem, True, opt.use_mem_select_num)
        print('test without memory:')
        logger.info('test without memory:')
        validate(datasets.target_valid_loader, device, model, running_metrics_val_wo_mem, False, opt.use_mem_select_num)

    if opt.stage != 'warm_up':
        score_w, class_iou_w = running_metrics_val_w_mem.get_scores()
        for k, v in score_w.items():
            print(k, v)
            logger.info('{}: {}'.format(k, v))
        for k, v in class_iou_w.items():
            logger.info('{}: {}'.format(k, v))
        running_metrics_val_w_mem.reset()

    score_wo, class_iou_wo = running_metrics_val_wo_mem.get_scores()
    for k, v in score_wo.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))
    for k, v in class_iou_wo.items():
        logger.info('{}: {}'.format(k, v))
    running_metrics_val_wo_mem.reset()

    torch.cuda.empty_cache()

    # save checkpoint
    state = {}
    if opt.stage != 'warm_up':
        state[model.nets[0].__class__.__name__] = {
            "model_state": model.nets[0].state_dict(),
            # "optimizer_state": model.optimizers[0].state_dict(),
            # "scheduler_state": model.schedulers[0].state_dict(),
            "objective_vectors": model.objective_vectors,
            "mem": model.mem
        }
    else:
        state[model.nets[0].__class__.__name__] = {
            "model_state": model.nets[0].state_dict(),
            # "optimizer_state": model.optimizers[0].state_dict(),
            # "scheduler_state": model.schedulers[0].state_dict(),
            "objective_vectors": model.objective_vectors
        }
    state[model.nets[1].__class__.__name__] = {
        "model_state": model.nets[1].state_dict(),
        # "optimizer_state": model.optimizers[0].state_dict(),
        # "scheduler_state": model.schedulers[0].state_dict()
    }
    if opt.stage != 'warm_up':
        state[model.nets[2].__class__.__name__] = {
            "model_state": model.nets[2].state_dict(),
            # "optimizer_state": model.optimizers[1].state_dict(),
            # "scheduler_state": model.schedulers[1].state_dict()
        }
    state['iter'] = iters + 1
    if opt.stage != 'warm_up':
        state['best_iou_w_mem'] = score_w["Mean IoU : \t"]
    state['best_iou_wo_mem'] = score_wo["Mean IoU : \t"]
    save_path = os.path.join(opt.logdir,"iter_{}.pth".format(iters+1))
    torch.save(state, save_path)

    # update best iou
    if opt.stage != 'warm_up':
        if score_w["Mean IoU : \t"] > model.best_iou_w_mem:
            model.best_iou_w_mem = score_w["Mean IoU : \t"]
    if score_wo["Mean IoU : \t"] > model.best_iou_wo_mem:
        model.best_iou_wo_mem = score_wo["Mean IoU : \t"]

def validate(valid_loader, device, model, running_metrics_val, use_mem, select_num):
    for data_i in tqdm(valid_loader):
        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)

        if use_mem:
            out = model.BaseNet_DP(images_val, mem=model.mem, select_num=select_num, aux_decoder=model.AuxDecoder_DP)
        else:
            out = model.BaseNet_DP(images_val, mem=None, select_num=select_num, aux_decoder=None)

        outputs = F.interpolate(out['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)
        #val_loss = loss_fn(input=outputs, target=labels_val)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()
    opt = relative_path_to_absolute_path(opt)
    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    # save config
    with open(os.path.join(opt.logdir, 'config_{}.txt').format(os.path.basename(opt.name)), 'w') as file:
        for k, v in vars(opt).items():
            file.write('{} : {}\n'.format(k, v))
    file.close()
    print('Save config successfully!')

    # save code
    project_name = os.path.basename(os.getcwd())
    file_paths = list()
    for root, directories, files in os.walk(os.getcwd()):
        for filename in files:
            filepath = os.path.join(root.replace(os.getcwd(), '') if root.replace(os.getcwd(), '')=='' else root.replace(os.getcwd(), '')[1:], filename)
            if filepath.endswith('.py') or filepath.endswith('.md') or filepath.endswith('.txt') or filepath.startswith(os.path.join(os.getcwd(), '.idea')):
                file_paths.append(filepath)
    with ZipFile(os.path.join(opt.logdir, '{}_{}.zip').format(project_name, os.path.basename(opt.name)), 'w') as zip:
        for file in file_paths:
            zip.write(file)
    zip.close()
    print('Save code successfully!')

    train(opt, logger)

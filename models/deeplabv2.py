# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.autograd import Variable
import torch.distributions as tdist
import mmcv
import copy
import pandas as pd

from scipy.stats import multivariate_normal
from scipy.stats import entropy
from scipy.stats import gaussian_kde

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine=affine_par)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate = 0.1, use_se = True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))
 
        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out

class ResNet101(nn.Module):
    def __init__(self, block, layers, num_classes, BatchNorm, bn_clr=False):
        self.num_classes = num_classes
        self.inplanes = 64
        self.bn_clr = bn_clr
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, BatchNorm=BatchNorm)
        # self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer5 = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        if self.bn_clr:
            self.bn_pretrain = BatchNorm(2048, affine=affine_par)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion, affine=affine_par))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, mem=None, select_num=5, entropy_threshold_ratio_use=0.6, aux_decoder=None, ssl=False, lbl=None,weight=False):
        out = dict()

        _, _, h, w = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out.update({'feat0':x})

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.bn_clr:
            x = self.bn_pretrain(x)

        out.update({'pre_compensate_feat': x})

        if mem:
            with torch.no_grad():
                if self.training:
                    x_target = x[int(x.shape[0]/2):x.shape[0]]
                    init_seg_out = aux_decoder(x_target.detach())
                    feat_mem_out = self.use_mem(x_target, init_seg_out, out['feat0'][int(x.shape[0]/2):x.shape[0]], mem, select_num, entropy_threshold_ratio_use, weight)
                    x = torch.add(x, torch.cat([torch.zeros_like(feat_mem_out), feat_mem_out]))
                else:
                    init_seg_out = aux_decoder(x.detach())
                    feat_mem_out = self.use_mem(x, init_seg_out, out['feat0'], mem, select_num, entropy_threshold_ratio_use, weight)
                    x = torch.add(x, feat_mem_out)

        out.update({'feat':x})
        aspp_out = self.layer5(x)
        out.update({'out':aspp_out})
        
        # if not ssl:
        #     x = nn.functional.upsample(x, (h, w), mode='bilinear', align_corners=True)
        #     if lbl is not None:
        #         self.loss = self.CrossEntropy2d(x, lbl)    
        # out['out'] = x
        return out

    def get_1x_lr_params(self):

        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())    
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]
    
    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10  
            
    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss

    def use_mem(self, fea, seg, input, mem, use_mem_select_num, entropy_threshold_ratio_use, weight=False):
        with torch.no_grad():
            N_class = 19
            class_entropy_hight_weight_avg = False
            class_entropy_low_weight_avg = True
            style_weight = False
            
            seg_norm = F.softmax(seg, dim=1)
            seg_sort_idx = seg_norm.argsort(dim=1, descending=True)[:, :N_class, :, :] # 降序
            class_mask = torch.zeros_like(seg_norm).scatter(1, seg_sort_idx, 1)
            class_mask_weight = torch.mul(class_mask, seg_norm)
            class_mask = torch.div(class_mask, class_mask.sum(dim=1).unsqueeze(1))
            class_mask_weight = torch.div(class_mask_weight, class_mask_weight.sum(dim=1).unsqueeze(1))

            seg_entropy = -1.0 * torch.sum(torch.mul(seg_norm, torch.div(torch.log(seg_norm), torch.tensor(np.log(2)))),dim=1)
            seg_entropy_line = seg_entropy.reshape(seg_entropy.shape[0], int(seg_entropy.shape[1]*seg_entropy.shape[2]))
            if entropy_threshold_ratio_use == 0:
                entropy_threshold, _ = torch.kthvalue(seg_entropy_line, 1, dim=1)
            else:
                entropy_threshold, _ = torch.kthvalue(seg_entropy_line, int(seg_entropy_line.shape[1] * entropy_threshold_ratio_use), dim=1)

            key_fea = self.cal_key(fea, input)
            fea_mem_out = torch.zeros_like(fea)

            BS, C_fea, H, W = fea.shape
            C_key = key_fea[0].shape[1]
            len_mem = mem['mu'].shape[1]

            for k in range(self.num_classes):
                # key_fea
                mu_fea = key_fea[0].reshape(BS, C_key, H * W)
                mu_fea = torch.repeat_interleave(mu_fea, len_mem, dim=2)
                sigma_fea = key_fea[1].reshape(BS, C_key, H * W)
                sigma_fea = torch.repeat_interleave(sigma_fea, len_mem, dim=2)

                # key_mem
                mu_mem = mem['mu'][k].permute(1, 0).unsqueeze(0).repeat(BS, 1, H * W)
                sigma_mem = mem['sigma'][k].permute(1, 0).unsqueeze(0).repeat(BS, 1, H * W)

                # calculate distance
                distance = self.wasserstein(mu_fea, sigma_fea, mu_mem, sigma_mem)
                distance = distance.reshape(BS, 1, H, W, len_mem)

                # 选取最近的use_mem_select_num个表项
                assert use_mem_select_num <= len_mem
                if use_mem_select_num == 0:
                    continue
                distance_kth_small = torch.topk(
                    distance, use_mem_select_num, dim=-1, largest=False, sorted=True).values[:, :, :, :, -1]
                distance_kth_small = distance_kth_small.unsqueeze(-1).repeat(1, 1, 1, 1, len_mem)
                if style_weight:
                    mask_val = torch.where(
                        distance <= distance_kth_small,
                        1.0/distance,
                        torch.zeros_like(distance))
                else:
                    mask_val = torch.where(
                        distance <= distance_kth_small,
                        torch.ones_like(distance),
                        torch.zeros_like(distance))

                if torch.sum(torch.sum(mask_val, dim=-1)) == 0:
                    continue

                # 对其val取平均
                val_mem = mem['val'][k].permute(1, 0).unsqueeze(0).repeat(BS, 1, 1)
                mask_val_tmp = mask_val.reshape(BS, H*W, len_mem).transpose(1,2)
                fea_mem_out_tmp = torch.div(
                    torch.bmm(val_mem, mask_val_tmp).reshape(BS, C_fea, H, W),
                    torch.sum(mask_val, dim=-1)) # 该类别的补偿fea

                fea_mem_out_tmp = torch.where(
                    seg_entropy.unsqueeze(1).repeat(1, C_fea, 1, 1) >= entropy_threshold.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, C_fea, H, W),
                    torch.mul(fea_mem_out_tmp, class_mask_weight[:,k:k+1]) if class_entropy_hight_weight_avg else torch.mul(fea_mem_out_tmp, class_mask[:,k:k+1]),
                    torch.mul(fea_mem_out_tmp, class_mask_weight[:, k:k + 1]) if class_entropy_low_weight_avg else torch.mul(fea_mem_out_tmp, class_mask[:, k:k + 1]))

                fea_mem_out = torch.add(fea_mem_out, fea_mem_out_tmp)
            return fea_mem_out

    def cal_key(self, fea, input):
        '''key=(mu, sigma)'''
        with torch.no_grad():
            BS, C_fea, H, W = fea.shape
            _, C_input, H_input, W_input = input.shape
            stride = int(H_input / H)
            assert H_input == H * stride and W_input == W * stride

            mu = torch.nn.AvgPool2d(stride)(input)
            sigma = torch.zeros_like(mu)
            for i in range(stride * stride):
                mask = torch.zeros(1, 1, stride, stride).cuda(input.device)
                mask[0, 0, int(i / stride), i % stride] = 1
                mask = mask.repeat(BS, C_input, H, W)
                x = torch.mul(input, mask)
                x = torch.nn.MaxPool2d(stride)(x)
                sigma = torch.add(sigma, torch.pow(torch.sub(x, mu), 2))
            sigma = torch.div(sigma, stride * stride - 1)

            return (mu, sigma)

    def wasserstein(self, mu1, sigma1, mu2, sigma2):
        p1 = torch.pow(torch.sub(mu1, mu2), 2)
        p2 = torch.pow(torch.sub(torch.pow(sigma1, 1 / 2), torch.pow(sigma2, 1 / 2)), 2)
        distance = torch.sum(torch.add(p1, p2), dim=1, keepdim=True)
        distance = torch.where(distance.isinf(), torch.full_like(distance, fill_value=1e38), distance)
        distance = torch.where(distance.isnan(), torch.full_like(distance, fill_value=1e38), distance)
        return distance

    def kld(self, mu1, sigma1, mu2, sigma2):
        gauss1 = tdist.Normal(mu1, sigma1)
        gauss2 = tdist.Normal(mu2, sigma2)
        distance = tdist.kl_divergence(gauss1, gauss2)
        distance = torch.sum(distance, dim=1, keepdim=True)
        distance = torch.where(distance.isinf(), torch.full_like(distance, fill_value=1e38), distance)
        distance = torch.where(distance.isnan(), torch.full_like(distance, fill_value=1e38), distance)
        return distance

    def visualize_seg_fea(self, fea):
        '''fea: 19*h*w'''
        palette = np.array(
            [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)],
            dtype=np.uint8)
        seg = fea.argmax(dim=0).cpu()
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]  # convert to BGR
        mmcv.imwrite(color_seg, 'work_dirs/test/hhh.jpg')

def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d)\
        or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def Deeplab(BatchNorm, num_classes=19, freeze_bn=False, restore_from=None, initialization=None, bn_clr=False):
    model = ResNet101(Bottleneck, [3, 4, 23, 3], num_classes, BatchNorm, bn_clr=bn_clr)
    if freeze_bn:
        model.apply(freeze_bn_func)
    if initialization is None:
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    else:
        pretrain_dict = torch.load(initialization)['state_dict']
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)

    if restore_from is not None:
        checkpoint = torch.load(restore_from)
        model.load_state_dict(checkpoint['ResNet101']["model_state"], strict=True)

        # # for DACS ckpt
        # model.load_state_dict(checkpoint['model'], strict=True)

    return model

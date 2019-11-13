import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import torch.nn.functional as F
import yaml
from torch.optim import lr_scheduler
import torch.nn.init as init
import random
from torchvision.utils import save_image, make_grid
import math

import os
import torchvision.utils as vutils
import pdb


def save_img(print_list, name, index, results_dir):
    # pdb.set_trace()
    nrow = len(print_list)
    img = torch.cat(print_list, dim=3)
    
    directory = os.path.join(results_dir, name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, '{:04d}'.format(index) + '.jpg')
    img = img.permute(1,0,2,3).contiguous()
    vutils.save_image(img.view(1,img.size(0),-1,img.size(3)).data, path, nrow=nrow, padding=0, normalize=True)

def ges_Aonfig(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def get_scheduler(optimizer, config, iterations=-1):
    if 'LR_POLICY' not in config or config['LR_POLICY'] == 'constant':
        scheduler = None # constant scheduler
    elif config['LR_POLICY'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['STEP_SIZE'],
                                        gamma=config['GAMMA'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config['LR_POLICY'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

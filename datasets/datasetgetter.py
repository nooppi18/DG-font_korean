import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap, CrossdomainFolder
import numpy as np
import pickle

class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img


def get_dataset(args):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])

    transform_val = Compose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.ToTensor(),
                                       normalize])

    class_to_use = args.att_to_use

    print('USE CLASSES', class_to_use)

    # remap labels
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1

    print("LABEL MAP:", remap_table)


    img_dir = args.data_dir

    # same data, different transformation
    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table)
    valdataset = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table)

    # parse classes to use
    tot_targets = torch.tensor(dataset.targets) # class(style) index
    T, V = np.unique(tot_targets, return_counts=True)
    c_num = V[0]

    min_data = 99999999
    max_data = 0

    train_idx = None
    val_idx = None
    torch.manual_seed(0)
    args.V_idx = torch.randperm(int(c_num))[:args.val_num] #[8, 33, 75, 97, 158, 199, 214, 183, 176, 83, 0, 44, 168, 24, 3]
    args.T_idx = [s for s in range(int(c_num)) if s not in args.V_idx]


    ## Saving Validation Contents into txt file
    # with open('font_freq_train/char_idx.pkl', 'rb') as f:
    #     font = pickle.load(f)
    # val_font = []
    # for idx in args.V_idx:
    #     val_font.append(font[int(idx)])
    
    # with open('val_content.txt', 'w') as fp:
    #     for font in val_font:
    #         fp.write('%s\n' %font)
    #     print('DONE')

    # assert False
        

    
    for k in class_to_use:
        tmp_idx = (tot_targets == k).nonzero()
        train_tmp_idx = tmp_idx[args.T_idx] #[:-args.val_num]#
        val_tmp_idx = tmp_idx[args.V_idx] #[-args.val_num:]#
        if k == class_to_use[0]:
            train_idx = train_tmp_idx.clone()
            val_idx = val_tmp_idx.clone()
        else:
            train_idx = torch.cat((train_idx, train_tmp_idx))
            val_idx = torch.cat((val_idx, val_tmp_idx))
        if min_data > len(train_tmp_idx):
            min_data = len(train_tmp_idx)
        if max_data < len(train_tmp_idx):
            max_data = len(train_tmp_idx)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(valdataset, val_idx)

    args.min_data = min_data
    args.max_data = max_data
    print("MINIMUM DATA :", args.min_data)
    print("MAXIMUM DATA :", args.max_data)

    train_dataset = {'TRAIN': train_dataset, 'FULL': dataset}

    return train_dataset, val_dataset



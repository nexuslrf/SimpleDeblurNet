import torch
import argparse
from network import SRNDeblurNet
from data import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,set_requires_grad,compute_psnr
import torchvision.transforms as transforms
from time import time
import PIL.Image as Image
import os

gpu_id = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('resume')
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--resume_epoch',default=None)
    parser.add_argument('--output_dir', default='./Output/', type=str)
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()

    img_list = np.load('./test_dir.npy').tolist()
    dataset = TestDataset(img_list, path = '../../gopro_dataset/')
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = True)

    net = SRNDeblurNet().cuda(gpu_id)
    set_requires_grad(net,False)
    last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  ) 


    with torch.no_grad():
        for step , batch in enumerate( dataloader ):
            for k in batch:
                if k == 'dir':
                    continue
                batch[k] = batch[k].cuda(gpu_id)
                batch[k].requires_grad = False

            y256,h = net( batch['img256'] )
            outs = transforms.ToPILImage()((y256.squeeze(0).cpu()+1)/2)
            save_dir = args.output_dir + batch['dir'][0].split('/')[2] + '/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            outs.save(save_dir+batch['dir'][0].split('/')[-1])
            print('finish '+batch['dir'][0])

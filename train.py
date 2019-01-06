import torch
import torch.nn as nn
import torchvision
from network import SRNDeblurNet
import numpy as np
from log import TensorBoardX
from utils import *
import train_config as config
from data import Dataset
from functools import partial
from tqdm import tqdm
from time import time
import copy
import sys

gpu_id = 3
log10 = np.log(10)
MAX_DIFF = 2 
std = [0.2317 , 0.2266, 0.2302]
avgpool = nn.AvgPool2d(2,2)
def compute_loss(  db256 , batch  ):
    assert db256.shape[0] == batch['label256'].shape[0]
    db128 = avgpool(db256); db64 = avgpool(db128);
    sp128 = avgpool(batch['label256']); sp64 = avgpool(sp128);
   # disp = torch.mean(db256-batch['label256']) 
    MSE = mse( db256 , batch['label256'] ) 
    loss = MSE + mse(db128,sp128) + mse(db64,sp64)
    psnr = 10*torch.log( MAX_DIFF**2 / MSE ) / log10
    return  {'mse':MSE , 'psnr':psnr, 'loss':loss}

def backward(loss , optimizer):
    optimizer.zero_grad()
    loss['mse'].backward()
    # torch.nn.utils.clip_grad_norm_( net.module.convlstm.parameters() , 3 )
    optimizer.step()
    return

def set_learning_rate(optimizer , epoch ):
    optimizer.param_groups[0]['lr'] = config.train['learning_rate']*config.train['decay_rate']**(epoch//config.train['lr_step'])

if __name__ == "__main__":
    tb = TensorBoardX(config_filename = 'train_config.py' , sub_dir = config.train['sub_dir'] )
    log_file = open('{}/{}'.format(tb.path,'train.log'),'w')

    train_img_list = np.load(config.train['train_img_list']).tolist()
    val_img_list = np.load(config.train['val_img_list']).tolist()

    train_dataset = Dataset( train_img_list, path = config.train['dataset_dir'])
    val_dataset = Dataset( val_img_list, path = config.train['dataset_dir'])
    train_dataloader = torch.utils.data.DataLoader(  train_dataset , batch_size = config.train['batch_size'] , shuffle = True , drop_last = True , num_workers = 8 , pin_memory = True) 
    val_dataloader = torch.utils.data.DataLoader(  val_dataset , batch_size = config.train['val_batch_size'] , shuffle = True , drop_last = True , num_workers = 2 , pin_memory = True) 


    
    mse = torch.nn.MSELoss().cuda(gpu_id)
    net = SRNDeblurNet(xavier_init_all = config.net['xavier_init_all']).cuda(gpu_id)
    #net = SRNDeblurNet(xavier_init_all = config.net['xavier_init_all']).cuda(gpu_id)

    assert config.train['optimizer'] in ['Adam' , 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam( net.parameters() , lr = config.train['learning_rate']  ,  weight_decay = config.loss['weight_l2_reg']) 
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD( net.parameters() , lr = config.train['learning_rate'] , weight_decay = config.loss['weight_l2_reg'] , momentum = config.train['momentum'] , nesterov = config.train['nesterov'] )

    last_epoch = -1 

    if config.train['resume'] is not None:
        last_epoch = load_model( net , config.train['resume'] , epoch = config.train['resume_epoch']  ) 

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer( optimizer , net  , config.train['resume_optimizer'] , epoch = config.train['resume_epoch'])
        assert last_epoch == _

    

    
    
    #train_loss_epoch_list = []

    train_loss_log_list = []
    val_loss_log_list = []
    first_val = True

    t = time()
    #convlstm_params = net.module.convlstm.parameters()
    #net_params = net.module.parameters()
    best_val_psnr = 0 
    best_net = None
    best_optimizer = None
    for epoch in tqdm(range( last_epoch + 1  , config.train['num_epochs'] ) , file = sys.stdout):
        set_learning_rate(optimizer,epoch)
        tb.add_scalar( 'lr' , optimizer.param_groups[0]['lr'] , epoch*len(train_dataloader) , 'train')
        for step , batch in tqdm(enumerate( train_dataloader ) , total= len(train_dataloader) , file=sys.stdout , desc = 'training'):
            t_list = []
            for k in batch:
                batch[k] = batch[k].cuda(gpu_id)
                batch[k].requires_grad = False
            db256, _ = net( batch['img256'])
            loss = compute_loss( db256 , batch )


            backward(loss,optimizer)

            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            train_loss_log_list.append( { k:loss[k] for k in loss} )
            for k,v in loss.items():
                tb.add_scalar( k , v , epoch*len(train_dataloader) + step , 'train' )

        #validate and log
        if first_val or epoch % config.train['log_epoch'] == config.train['log_epoch'] -  1  :
            with torch.no_grad():
                first_val = False
                for step,batch in tqdm(enumerate(val_dataloader), total = len(val_dataloader) , file = sys.stdout ,desc = 'validating' ):
                    for k in batch:
                        batch[k] = batch[k].cuda(gpu_id)
                        batch[k].requires_grad = False
                    db256 , _ = net( batch['img256'] )
                    loss  = compute_loss( db256,  batch )
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    val_loss_log_list.append( { k:loss[k] for k in loss} )

                train_loss_log_dict = { k: float( np.mean( [ dic[k] for dic in train_loss_log_list] )) for k in train_loss_log_list[0] }
                val_loss_log_dict = { k: float( np.mean( [ dic[k] for dic in val_loss_log_list] )) for k in val_loss_log_list[0] }
                for k,v in val_loss_log_dict.items():
                    tb.add_scalar( k , v , (epoch+1)*len(train_dataloader) , 'val'  ) 
                if best_val_psnr < val_loss_log_dict['psnr']:
                    best_val_psnr = val_loss_log_dict['psnr']
                save_model( net , tb.path , epoch )
                save_optimizer( optimizer , net , tb.path , epoch )


                train_loss_log_list.clear() 
                val_loss_log_list.clear()

                tt = time()
                log_msg = ""
                log_msg += "epoch {} , {:.2f} imgs/s".format( epoch  , ( config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len( val_dataloader ) * config.train['val_batch_size'] ) / (tt - t) )

                log_msg += " | train : "
                for idx,k_v in enumerate(train_loss_log_dict.items()):
                    k,v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k,v,',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k,v,',')
                log_msg += "  | val : "
                for idx,k_v in enumerate(val_loss_log_dict.items()):
                    k,v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k,v,',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k,v,',' if idx < len(val_loss_log_list) - 1 else '')
                tqdm.write( log_msg , file = sys.stdout )
                sys.stdout.flush()
                log_file.write(log_msg+'\n')
                log_file.flush()
                t = time()
                    #print( torch.max( predicts , 1  )[1][:5] )

            #train_loss_epoch_list = [] 
                

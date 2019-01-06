#train_config
test_time = False

train = {}


train['train_img_list'] = './train_dir.npy'
train['val_img_list'] = './test_dir.npy'
train['dataset_dir'] = '../../gopro_dataset/'
train['batch_size'] = 32
train['val_batch_size'] = 32
train['num_epochs'] = 2400
train['lr_step'] = 600 
train['log_epoch'] = 10
train['optimizer'] = 'Adam'
train['learning_rate'] = 1e-4
train['decay_rate'] = 0.3

#-- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True 


#config for save , log and resume
train['sub_dir'] = 'RF_new'
train['resume'] =  './save/RF_new/0'
train['resume_epoch'] = None  #None means the last epoch
train['resume_optimizer'] = './save/RF_new/0'

net = {}
net['xavier_init_all'] = True

loss = {}
loss['weight_l2_reg'] = 0.0

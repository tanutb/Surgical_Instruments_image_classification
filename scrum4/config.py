##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 4                 # number of Dataloader workers
epochs = 100                 # number of epochs
batch_size = 4             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
net = 'inception_mixed_6e'  # feature extractor
num_attentions = 16          # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'sur'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models
save_dir = 'C:\\Users\\gmita\\Desktop\\work_3_2_2565\\onboard\\WS-DAN.PyTorch-master\\c'
model_name = 'model03.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = True
eval_ckpt = save_dir + "\\"+ model_name
eval_savepath = 'C:\\Users\\gmita\\Desktop\\work_3_2_2565\\onboard\\WS-DAN.PyTorch-master\\v'
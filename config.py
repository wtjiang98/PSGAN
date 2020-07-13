import os

from easydict import EasyDict as edict

default = edict()

default.snapshot_path = './snapshot/'
default.vis_path = './visulization/'
default.log_path = './log/'
default.data_path = '/data/makeup-transfer/face-style-with-MT-makeup'

config = edict()

config.task_name = ""

config.direct = True
config.cls_list = ["SYMIX", "MAKEMIX"]
config.norm = "SN"
config.content_layer = "r41"

# setting for cycleGAN
# Hyper-parameters

config.multi_gpu = False
config.gpu_ids = [0,1,2]

# Setting path
config.snapshot_path = default.snapshot_path
config.pretrained_path = default.snapshot_path
config.vis_path = default.vis_path
config.log_path = default.log_path
config.data_path = default.data_path
config.num_workers = 16

# Setting training parameters
config.G_LR = 2e-5
config.D_LR = 2e-5
config.beta1 = 0.5
config.beta2 = 0.999
config.c_dim = 2
config.num_epochs = 50
config.num_epochs_decay = 100
config.ndis = 1
config.snapshot_step = 260
config.log_step = 10
config.vis_step = config.snapshot_step
config.batch_size = 1
config.lambda_A = 10.0
config.lambda_B = 10.0
config.lambda_idt = 0.5
config.lambda_cls = 1
config.lambda_rec = 10
config.lambda_his = 1
config.lambda_skin = 0.1
config.lambda_eye = 1
config.lambda_his_lip = config.lambda_his
config.lambda_his_skin = config.lambda_his * config.lambda_skin
config.lambda_his_eye = config.lambda_his * config.lambda_eye
config.lambda_vgg = 5e-3

config.landmark_points = 68
config.img_size = 256
config.g_conv_dim = 64
config.d_conv_dim = 64
config.g_repeat_num = 6
config.d_repeat_num = 3

config.checkpoint = "assets/models"

config.test_model = "51_2000"

# Serving configs

config.master_url = "http://ae.wanzhaoyi.ws2.hh-b.brainpp.cn:5000"

# Preprocess configs
config.up_ratio = 0.6 / 0.85  # delta_size / face_size
config.down_ratio = 0.2 / 0.85  # delta_size / face_size
config.width_ratio = 0.2 / 0.85  # delta_size / face_size
config.lip_class = [7, 9]
config.face_class = [1, 6]


# Setting datasets
dataset_config = edict()

dataset_config.name = 'MAKEUP'
dataset_config.dataset_path = default.data_path
dataset_config.img_size = 256



def generate_config(_network, _dataset):
    for k, v in dataset_config[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v

def merge_cfg_arg(config, args):
    config.gpu_ids = [int(i) for i in args.gpus.split(',')]
    config.batch_size = args.batch_size
    config.vis_step = args.vis_step
    config.snapshot_step = args.vis_step
    config.ndis = args.ndis
    config.lambda_cls = args.lambda_cls
    config.lambda_A = args.lambda_rec
    config.lambda_B = args.lambda_rec
    config.G_LR = args.LR
    config.D_LR = args.LR
    config.num_epochs_decay = args.decay
    config.num_epochs = args.epochs
    config.num_workers = args.num_workers
    config.whichG = args.whichG
    config.task_name = args.task_name
    config.norm = args.norm
    config.lambda_his = args.lambda_his
    config.lambda_vgg = args.lambda_vgg
    config.cls_list = [i for i in args.cls_list.split(',')]
    config.content_layer = [i for i in args.content_layer.split(',')]
    config.direct = args.direct
    config.lips = args.lips
    config.skin = args.skin
    config.eye = args.eye
    config.g_repeat = args.g_repeat
    config.lambda_his_lip = args.lambda_his
    config.lambda_his_skin = args.lambda_his * args.lambda_skin
    config.lambda_his_eye = args.lambda_his * args.lambda_eye
    config.checkpoint = args.checkpoint
    print(config)
    if args.opts is not None:
        for key, value in args.opts:
            setattr(config, key, value)
    if "checkpoint" in config.items():
        config.checkpoint = args.checkpoint
    if "test_model" in config.items():
        config.test_model = args.test_model

    return config


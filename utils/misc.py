import sys, os, logging, shutil
from torch.utils.tensorboard import SummaryWriter   
import torch, random
import numpy as np
from collections import Counter
ALL_DATASETS=['APTOS','DEEPDR','FGADR','IDRID','MESSIDOR','RLDR']
ESDG_DATASETS = ['APTOS','DEEPDR','FGADR','IDRID','MESSIDOR','RLDR','DDR','EYEPACS']
ALL_METHODS = ['GDRNet', 'ERM', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen']

def count_samples_per_class(targets, num_classes):
    counts = Counter()
    for y in targets:
        counts[int(y)] += 1
    return [counts[i] if counts[i] else np.inf for i in range(num_classes)]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_log(args, cfg, log_path, train_loader_length, dataset_size):
    assert cfg.ALGORITHM in ALL_METHODS
    if not cfg.RANDOM:
        setup_seed(cfg.SEED)
        
    init_output_foler(cfg, log_path)
    writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
    writer.add_text('config', str(args))
    logging.basicConfig(filename=log_path + '/log.txt', level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(train_loader_length))
    logging.info("We have {} images in train set, {} images in val set, and {} images in test set.".format(dataset_size[0], dataset_size[1], dataset_size[2]))
    logging.info(str(args))
    logging.info(str(cfg))
    return writer

def init_output_foler(cfg, log_path):
    if os.path.isdir(log_path):
        if cfg.OVERRIDE:
            shutil.rmtree(log_path)
        else:
            if os.path.exists(os.path.join(log_path, 'done')):
                print('Already trained, exit')
                exit()
            else:
                shutil.rmtree(log_path)
    else:
        os.makedirs(log_path)

def get_scheduler(optimizer, max_epoch):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch * 0.5], gamma=0.1)
    return scheduler

def update_writer(writer, epoch, scheduler, loss_avg):
    logging.info('epoch: {}, total loss: {}'.format(epoch, loss_avg.mean()))
    writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch) 
    writer.add_scalar('info/loss', loss_avg.mean(), epoch)

class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data
    
class LossCounter:
    def __init__(self, start = 0):
        self.sum = start
        self.iteration = 0
    def update(self, num):
        self.sum += num
        self.iteration += 1
    def mean(self):
        return self.sum * 1.0 / self.iteration
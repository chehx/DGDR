import algorithms
import os
from utils.validate import *
from utils.args import *
from utils.misc import *
from dataset.data_manager import get_dataset
from tqdm import tqdm

if __name__ == "__main__":

    args = get_args()
    cfg = setup_cfg(args)
    log_path = os.path.join('./result/fundusaug', cfg.OUTPUT_PATH)
       
    # init
    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    writer = init_log(args, cfg, log_path, len(train_loader), dataset_size)
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.cuda()
    
    # train
    iterator = tqdm(range(cfg.EPOCHS))
    scheduler = get_scheduler(algorithm.optimizer, cfg.EPOCHS)
    best_performance = 0.0
    for i in iterator:
        
        epoch = i + 1
        loss_avg = LossCounter()
        for image, mask, label, domain, img_index in train_loader:

            algorithm.train()
            minibatch = [image.cuda(), mask.cuda(), label.cuda().long(), domain.cuda().long()]
            loss_dict_iter = algorithm.update(minibatch)   
            loss_avg.update(loss_dict_iter['loss'])

        alpha = algorithm.update_epoch(epoch)
        update_writer(writer, epoch, scheduler, loss_avg)
        scheduler.step()

        # validation
        if epoch % cfg.VAL_EPOCH == 0:
            val_auc, test_auc = algorithm.validate(val_loader, test_loader, writer)
            if val_auc > best_performance and epoch > cfg.EPOCHS * 0.3:
                best_performance = val_auc
                algorithm.save_model(log_path)
    
    algorithm.renew_model(log_path)
    _, test_auc = algorithm.validate(val_loader, test_loader, writer)
    os.mknod(os.path.join(log_path, 'done'))
    writer.close()

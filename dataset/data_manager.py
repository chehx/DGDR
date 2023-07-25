from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataset(cfg):
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)

    batch_size = cfg.BATCH_SIZE
    drop_last = cfg.DROP_LAST
    num_worker = min (batch_size // 4, 4)

    train_dataset = GDRBench(root = cfg.DATASET.ROOT, source_domains= cfg.DATASET.SOURCE_DOMAINS, target_domains = cfg.DATASET.TARGET_DOMAINS, \
                                mode = 'train', trans_basic=train_ts, trans_mask=tra_fundus)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_worker, drop_last=drop_last, pin_memory=True)

    val_dataset = GDRBench(root = cfg.DATASET.ROOT, source_domains= cfg.DATASET.SOURCE_DOMAINS, target_domains = cfg.DATASET.TARGET_DOMAINS, \
                                mode = 'val', trans_basic=test_ts)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)

    test_dataset = GDRBench(root = cfg.DATASET.ROOT, source_domains= cfg.DATASET.SOURCE_DOMAINS, target_domains = cfg.DATASET.TARGET_DOMAINS, \
                                mode = 'test', trans_basic=test_ts)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)        

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]
    print('dataset size:', dataset_size)
    return train_loader, val_loader, test_loader, dataset_size

def get_transform(cfg):

    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_train = transforms.Compose([
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize,
    ])
    
    tra_test = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    tra_mask= transforms.Compose([
                transforms.Resize(re_size),
                transforms.ToTensor()])
    
    return tra_train, tra_test, tra_mask


# FundusAug contains ["RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "Sharpness", "Halo", "Hole", "Spot", "Blur"],
# these operations are splited into pre_FundusAug and post_FundusAug

def get_pre_FundusAug(cfg):

    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_train = transforms.Compose([
        transforms.Resize(size),
        transforms.ColorJitter( brightness=cfg.TRANSFORM.COLORJITTER_B, \
                                contrast=cfg.TRANSFORM.COLORJITTER_C, \
                                saturation=cfg.TRANSFORM.COLORJITTER_S, \
                                hue=cfg.TRANSFORM.COLORJITTER_H),
        transforms.ToTensor()
    ])
    
    tra_test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(re_size),
            transforms.ToTensor(),
            normalize])
    
    tra_mask= transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()])
    
    return tra_train, tra_test, tra_mask

def get_post_FundusAug(cfg):
    aug_prob = cfg.TRANSFORM.AUGPROB
    size = 256
    re_size = 224
    normalize = get_normalize()

    tra_fundus_1 = FundusAug.Compose([
        FundusAug.Sharpness(prob = aug_prob),
        FundusAug.Halo(size, prob=aug_prob),
        FundusAug.Hole(size, prob=aug_prob),
        FundusAug.Spot(size, prob=aug_prob),
        FundusAug.Blur(prob=aug_prob)
    ])
    
    tra_fundus_2 = transforms.Compose([
                transforms.RandomCrop(re_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                normalize])
    
    return {'post_aug1':tra_fundus_1, 'post_aug2':tra_fundus_2}

def get_normalize():

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    return normalize
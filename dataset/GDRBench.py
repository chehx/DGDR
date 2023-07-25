import os.path as osp
from torch.utils.data.dataset import Dataset
from PIL import Image

# Dataset for fundus images including APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR (and DDR, Eyepacs for ESDG)
class GDRBench(Dataset):

    def __init__(self, root, source_domains, target_domains, mode, trans_basic=None, trans_mask = None, trans_fundus=None):

        root = osp.abspath(osp.expanduser(root))
        self.mode = mode
        self.dataset_dir = osp.join(root, "images")
        self.split_dir = osp.join(root, "splits")

        self.data = []
        self.label = []
        self.domain = []
        self.masks = []
        
        self.trans_basic = trans_basic
        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask

        if mode == "train":
            self._read_data(source_domains, "train")
        elif mode == "val":
            self._read_data(source_domains, "crossval")
        elif mode == "test":
            self._read_data(target_domains, "test")
        
    def _read_data(self, input_domains, split):
        items = []
        for domain, dname in enumerate(input_domains):
            if split == "test":
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval.txt")
                impath_label_list += self._read_split(file_val)
            else:
                file = osp.join(self.split_dir, dname + "_" + split + ".txt")
                impath_label_list = self._read_split(file)

            for impath, label in impath_label_list:
                self.data.append(impath)
                self.masks.append(impath.replace("images", "masks"))

                self.label.append(label)
                self.domain.append(domain)

    def _read_split(self, split_file):
        items = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))
                
        return items

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data = Image.open(self.data[index]).convert("RGB")

        if self.mode == "train":
            mask = Image.open(self.masks[index]).convert("L")

        label = self.label[index]
        domain = self.domain[index]
        
        if self.trans_basic is not None:
            data = self.trans_basic(data)
        
        if self.trans_mask is not None:
            mask = self.trans_mask(mask)

        if self.mode == "train":
            return data, mask, label, domain, index
        else:
            return data, label, domain, index
    


from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 


class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 vit_mode = "h"):
        
        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
        """
        
        self.data_root_dir = data_root_dir 
        self.mode = mode 
        self.vit_mode = vit_mode
        
        # directory containing all binary annotations
        self.mask_dir = osp.join(self.data_root_dir, self.mode, "binary_annotations")

        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.data_root_dir, self.mode, f"sam_features_{self.vit_mode}", mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        return sam_feat, mask_name, cls_id
 

class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h"):
                        
        self.data_root_dir = data_root_dir
        self.vit_mode = vit_mode
        
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}
        
                
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]

        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.data_root_dir, "binary_annotations", f"seq{seq}")
            self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.data_root_dir, f"sam_features_{self.vit_mode}", mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        return sam_feat, mask_name, cls_id
    

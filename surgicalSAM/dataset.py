from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 
import cv2 
import pdb
from sklearn.utils import class_weight
import torch

def find_masks(data_root_dir, l_names, version):

    mask_list=[]
    class_list = []
    for name in l_names: ## video name
        if os.path.isdir(os.path.join(data_root_dir, name)):
            vid_dir = os.path.join(data_root_dir, name)

            for frame_n in os.listdir(vid_dir):

                mask_dir = os.path.join(vid_dir, frame_n, str(version),'binary_annotations')

                if os.path.isdir(mask_dir):
                    for f in os.listdir(mask_dir):
                        if os.path.splitext(f)[1] == '.png':

                            if 'class' in f:
                                file_version = os.path.join(mask_dir, f)
                                if os.path.exists(file_version):
                                    mask_list.append(file_version)
                                    class_name = os.path.splitext(f.split('class')[1])[0]
                                    class_list.append(int(class_name))
    return mask_list, class_list


class HysterectomyDataset(Dataset):
    def __init__(self, data_root_dir = "../hysterectomy/Clips/", 
                 mode = "val", 
                 vit_mode = "h",
                 version = 1, 
                 img_size=256,
                 num_classes=1):
        
        """Define the hysterectomy dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for hysterectomy
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """
        
        self.vit_mode = vit_mode
        self.version = version
        self.img_size = img_size
        self.num_classes = num_classes

       
        # directory containing all binary annotations
        if mode == "train":
            list_videos = ['Hyst_BB_1.20.23b', 'Hyst_SurgU_3.21.23a', 'AutoLaparo']
        
        elif mode == "val":
            list_videos = ['Hyst_JS_1.30.23','Hyst_BB_4.14.23']

        self.mask_list, self.class_list =  find_masks(data_root_dir, list_videos, version)

        # compute weights and return

        # self.mask_list = self.mask_list[:50]


    def __len__(self):
        return len(self.mask_list)
    
    def dataset_weights(self):


        unique_classes = np.sort(np.unique(self.class_list))
        unique_class_weights = np.array(class_weight.compute_class_weight(  class_weight='balanced',
                                                                            classes=unique_classes, 
                                                                            y=self.class_list))
        
        print(unique_class_weights)
        return torch.Tensor(unique_class_weights)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]


        sub_dir, mask_id = mask_name.split(f'/{self.version}/binary_annotations')
        frame_n = sub_dir.split('/')[-1]

        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        # if cls_id == 0:
        #     cls_id = self.num_classes
        # cls_id = 1

        # get pre-computed sam feature 
        feat_dir = osp.join(sub_dir, str(self.version), f"sam_features_{self.vit_mode}", frame_n + '.npy')
        try:
            sam_feat = np.load(feat_dir)
        except:
            sam_feat = np.zeros((64, 64, self.img_size))    
        # get ground-truth mask
        try: 
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size)) 
        except:
            mask = np.zeros((self.img_size, self.img_size))
        
        # get class embedding
        class_embedding_path = mask_name.replace("binary_annotations", f"class_embeddings_{self.vit_mode}").replace("png","npy")
        
        try:
            class_embedding = np.load(class_embedding_path)
        except:
            class_embedding = np.zeros((self.img_size, ))
            
        return sam_feat, mask_name, cls_id, mask, class_embedding
 

class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 vit_mode = "h",
                 version = 0):
        
        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """

        self.img_size=256
        
        self.vit_mode = vit_mode
       
        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")

        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            for f in files:
                self.mask_list.append(osp.join(subdir, f))

        print(len(self.mask_list))
        
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]


        # cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        cls_id = 1

        mask_dir = mask_name.split('binary_annotations/')[1]
        feat_name = mask_dir.split('_class')[0] + '.npy'
        embedding_name = os.path.splitext(mask_dir)[0] + '.npy'

        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), feat_name)
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size)) 


        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), embedding_name)
        class_embedding = np.load(class_embedding_path)

        return sam_feat, mask_name, cls_id, mask, class_embedding
 

class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h",
                 version = 0):
                        
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

        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        
        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        
        return sam_feat, mask_name, cls_id, mask, class_embedding
    

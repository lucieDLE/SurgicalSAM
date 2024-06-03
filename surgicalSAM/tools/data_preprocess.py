import sys
sys.path.append("../..")
import os
import os.path as osp
import cv2
import numpy as np
from segment_anything_Surg import sam_model_registry, SamPredictor
from segment_anything_Surg.utils.transforms import ResizeLongestSide
from PIL import Image
import random 
import torch 
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.nn import functional as F
import argparse 

from tqdm import tqdm
import pdb 

def augmentation(image, masks, scale_factor, rotate_angle, colour_factor, H, W, scale = False, rotate = False, colour = False):
    """Generate augmentation to image and masks
       image - original image
       masks - binary masks for all the classes present in the image (list)
       scale_factor - how much to scale the image and the masks
       rotate_angle - rotation angle value in degrees on the image and the masks
       colour_factor - how much to jitter the brightness, contrast, and satuation of the image
       H - height of original image
       W - width of original image
       scale - whether to adjust scale (bool)
       rotate - whether to rotate (bool)
       colour - whether to adjust colour (bool)

    Returns:
        image - image after the augmentation
        masks - masks after the augmentation (list)
    """
    
    # Scale and crop
    if scale:
        # Scale
        if random.random() > 0.5:
            scale = random.random()*scale_factor + 1
            resize = transforms.Resize(size=(int(H*scale), int(W*scale)))
            image = resize(image) 
            masks = [resize(mask) for mask in masks]

            # Crop
            i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(H, W))
            image = TF.crop(image, i, j, h, w) 
            masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        
    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        masks = [TF.hflip(mask) for mask in masks]

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        masks = [TF.vflip(mask) for mask in masks]
    
    # Rotate 
    if rotate:
        if random.random() > 0.5:
            angle = rotate_angle * random.random() * (random.random()>0.5)
            image = TF.rotate(image, angle)
            masks = [TF.rotate(mask, angle) for mask in masks]
    
    # Colour jitter
    if colour:
        if random.random() > 0.5:
            colour_jitter = transforms.ColorJitter(brightness=colour_factor, contrast=colour_factor, saturation=colour_factor)
            image = colour_jitter(image)
            
    return image, masks


def version_to_augmentation_toggles(version, n_version):
    """Generate the toggles for scale, rotate, and colour for different augmentation versions
       version - current version 
       n_version - total number of versions 
       
       The augmentation settings for different versions (if n_version == 40):
       Version | Flip | Scale | Rotate | Colour 
       1 - 10     √       x       x         x
       11 - 20    √       √       x         x
       21 - 30    √       x       √         x
       31 - 40    √       x       x         √

    """
    scale = False
    rotate = False
    colour = False 
    
    if (n_version//4) < version < ((2*n_version//4)+1):
        scale = True 
    elif (2*n_version//4) < version < ((3*n_version//4)+1):
        rotate = True 
    elif (3*n_version//4) < version < n_version+1:
        colour = True 
        
    return scale, rotate, colour


def set_mask(mask):   
    """ Transform the mask to the form expected by SAM, the transformed mask will be used to generate class embeddings
        Adapated from set_image in the official code of SAM https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py
    """
    input_mask = ResizeLongestSide(1024).apply_image(mask)
    input_mask_torch = torch.as_tensor(input_mask)
    input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_mask = set_torch_image(input_mask_torch)
    
    return input_mask


def set_torch_image(transformed_mask):
    input_mask = preprocess(transformed_mask)  # pad to 1024
    return input_mask


def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


# specify the dataset
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["endovis_2017", "endovis_2018", "hysterectomy"], help='which dataset to use')
parser.add_argument('--n-version', type=int, default=40, help='total number of augmentation versions to generate')
parser.add_argument('--data_root', type=str, default='../../data/', help='path to data directory')

args = parser.parse_args()


# define the SAM model 
vit_mode = "h"
if vit_mode == "h":
    sam_checkpoint = "../../ckp/sam/sam_vit_h_4b8939.pth"
sam = sam_model_registry[f"vit_{vit_mode}"](checkpoint=sam_checkpoint)
sam.cuda()
predictor = SamPredictor(sam)

# define data
dataset_name = args.dataset

if dataset_name == "endovis_2018":
    data_root_dir = f"../../data/{dataset_name}"

    mask_dir = osp.join(data_root_dir, "train", "0", "binary_annotations")
    frame_dir = osp.join(data_root_dir, "train",  "0", "images")

    frame_list = [os.path.join(os.path.basename(subdir), file) for subdir, _, files in os.walk(frame_dir) for file in files if files]
    mask_list = [os.path.join(os.path.basename(subdir), file) for subdir, _, files in os.walk(mask_dir) for file in files if files]

elif dataset_name == "endovis_2017":
    data_root_dir = f"../../data/{dataset_name}"

    mask_dir = osp.join(data_root_dir, "0", "binary_annotations")
    frame_dir = osp.join(data_root_dir, "0", "images")

    frame_list = [os.path.join(os.path.basename(subdir), file) for subdir, _, files in os.walk(frame_dir) for file in files if files]
    mask_list = [os.path.join(os.path.basename(subdir), file) for subdir, _, files in os.walk(mask_dir) for file in files if files]

elif dataset_name == "hysterectomy":
    data_root_dir = args.data_root
    mask_list, frame_list = [], []

    # for name in os.listdir(data_root_dir):
    for name in ['AutoLaparo']:
        
        if os.path.isdir(os.path.join(data_root_dir, name)):
            vid_dir = os.path.join(data_root_dir, name)
            print(f"processing {vid_dir}...")

            for frame_n in os.listdir(vid_dir):
                frame_path = os.path.join(vid_dir,frame_n, '0')
                
                mask_dir = os.path.join(frame_path, 'binary_annotations')
                if os.path.isdir(mask_dir):
                    img = os.path.join(frame_path, 'images', frame_n + '.jpg')
                    for f in os.listdir(mask_dir):
                        if os.path.splitext(f)[1] == '.png':
                            mask_list.append(os.path.join(mask_dir, f))
                            frame_list.append(img)


print(f"Found {len(frame_list)} frames with {len(mask_list)} masks")
# define augmentation factors 
scale_factor = 0.2
rotate_angle = 30
colour_factor = 0.4

H = 256
W = 256 

n_version = args.n_version

progress_bar =  tqdm(enumerate(frame_list))

# progress_bar.set_description(f"frame_name")

# go though each frame one by one
for n, frame_name in progress_bar:
    with torch.no_grad():
    
        # read the original frame (without any augmentation)
        # frame_path = osp.join(frame_dir, frame_name)
        frame_path = frame_name
        frame_dir, frame_name = os.path.split(frame_name)
        progress_bar.set_postfix_str(f"frame {n} / {len(frame_list)}")
        original_frame = cv2.imread(frame_path)

        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.resize(original_frame, (H,W))
        original_frame = Image.fromarray(original_frame)

        save_dir = osp.join(frame_dir.split('/0/')[0],'1')

        # if not os.path.exists(save_dir):
        
        # read all the original masks (without any augmentation) of the current frame and organise them into a list
        frame_idx = os.path.splitext(frame_name)[0]
        masks_name = []
        for mask in mask_list:
            
            mask_idx = os.path.split(mask)[1]
            mask_idx = mask_idx.split('_')[0]

            if mask_idx == frame_idx :
                masks_name.append(mask)

        masks_name = sorted(masks_name)

        original_masks = []
        for version in range(1, n_version+1):
            save_dir = osp.join(frame_dir.split('/0/')[0], str(version))


            if not os.path.exists(save_dir):
                print(save_dir)
                for mask_path in masks_name:
                    # mask_path = osp.join(mask_dir, mask_name)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
                    mask = np.uint8(mask == 255)
                    mask = cv2.resize(mask, (H,W))

                    mask = Image.fromarray(mask)
                    original_masks.append(mask)
                
                    # for each frame, generate 40 different versions of augmentation
                    # set seed for reproducibility
                    random.seed(version)
                    torch.manual_seed(version)
                    torch.cuda.manual_seed(version)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    np.random.seed(version)

                    # perform augmentation to the frame and its masks based on the current version number
                    scale, rotate, colour = version_to_augmentation_toggles(version, n_version)       
                    frame, masks = augmentation(original_frame, original_masks, scale_factor, rotate_angle, colour_factor,
                                                H, W, scale = scale, rotate = rotate, colour = colour)
                    frame = np.asarray(frame)

                    masks = [np.asarray(mask)*255 for mask in masks]
                    
                    # obtain SAM feature of the augmented frame
                    predictor.set_image(frame)
                    feat = predictor.features.squeeze().permute(1, 2, 0)
                    feat = feat.cpu().numpy()

                    # save the augmented frame and its SAM feature 
                    if dataset_name == "endovis_2018":
                        save_dir = osp.join(data_root_dir, "train", str(version))
                    elif dataset_name == "endovis_2017":
                        save_dir = osp.join(data_root_dir, str(version))

                    elif dataset_name == "hysterectomy":
                        dirname = frame_dir.split('/0/')[0]
                        save_dir = osp.join(dirname, str(version))

                    os.makedirs(save_dir, exist_ok = True) 


                    frame_save_dir = osp.join(save_dir,  "images", frame_name)
                    feat_save_dir = osp.join(save_dir, f"sam_features_{vit_mode}", frame_name.split(".")[0] + ".npy")
                    os.makedirs(osp.dirname(frame_save_dir), exist_ok = True) 
                    os.makedirs(osp.dirname(feat_save_dir), exist_ok = True) 
                    
                    frame = Image.fromarray(frame)
                    # print(frame.shape)
                    # frame = cv2.resize(frame, (H,W))

                    frame.save(frame_save_dir)
                    np.save(feat_save_dir, feat)
                    
                    # go through each augmented mask
                    for mask, mask_name in zip(masks, masks_name):
                        
                        # process augmented_masks to the same shape and format as the image 
                        mask_name = os.path.split(mask_name)[1]
                        zeros = np.zeros_like(mask)
                        mask_processed = np.stack((mask, zeros, zeros),axis=-1)
                        mask_processed = set_mask(mask_processed)   
                        mask_processed = F.interpolate(mask_processed, size=torch.Size([64, 64]), mode="bilinear")
                        mask_processed = mask_processed.squeeze()[0]
                        
                        # if the augmented mask after processing does not have any foreground objects, then skip this mask
                        if (True in (mask_processed > 0)) == False:
                            continue 
                        
                        # compute the class embedding using frame SAM feature and processed mask
                        class_embedding = feat[mask_processed > 0]
                        class_embedding = class_embedding.mean(0).squeeze()
                        
                        # save the augmented mask and the computed class embedding
                        mask_save_dir = osp.join(save_dir, "binary_annotations", mask_name)
                        class_embedding_save_dir = osp.join(save_dir, f"class_embeddings_{vit_mode}", mask_name.replace("png","npy"))
                        os.makedirs(osp.dirname(mask_save_dir), exist_ok = True) 
                        os.makedirs(osp.dirname(class_embedding_save_dir), exist_ok = True) 
                        
                        mask = Image.fromarray(mask)
                        mask.save(mask_save_dir)
                        np.save(class_embedding_save_dir, class_embedding)

import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re 

import pdb

def create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr):

    """Gather the predicted binary masks of different frames and classes into a dictionary, mask quality is also recorded

    Returns:
        dict: a dictionary containing all predicted binary masks organised based on sequence, frame, and mask name
    """

    preds = preds.cpu()
    
    preds_quality = preds_quality.cpu()
    
    pred_masks = (preds > thr).int()
    

    for pred_mask, mask_name, pred_quality in zip(pred_masks, mask_names, preds_quality):        
      
        seq_name = mask_name.split("/")[0]
        frame_name = osp.basename(mask_name).split("_")[0]
        
        if seq_name not in binary_masks.keys():
            binary_masks[seq_name] = dict()
        
        if frame_name not in binary_masks[seq_name].keys():
            binary_masks[seq_name][frame_name] = list()
            
        binary_masks[seq_name][frame_name].append({
            "mask_name": mask_name,
            "mask": pred_mask,
            "mask_quality": pred_quality.item()
        })
        
    return binary_masks
        
def create_hyst_binary_masks(binary_masks, preds, preds_quality, mask_names, thr):
    
    preds_quality = preds_quality.cpu()
    pred_masks = preds.cpu()
    
    for pred_mask, mask_name, pred_quality in zip(pred_masks, mask_names, preds_quality):        
      
        frame_number = os.path.split(mask_name)[1].split('_')[0]
        vid_name = mask_name.split('_Videos/')[1].split(frame_number)[0]

        if vid_name not in binary_masks.keys():
            binary_masks[vid_name] = dict()

        if frame_number not in binary_masks[vid_name].keys():
            binary_masks[vid_name][frame_number] = list()


        binary_masks[vid_name][frame_number].append({
               "mask_name": mask_name,
                    "mask": pred_mask,
            "mask_quality": pred_quality.item()
        })

    return binary_masks

def create_hysterectomy_masks(binary_masks, H, W, num_classes):
    """given the dictionary containing all predicted binary masks, compute final prediction of each frame and organise the prediction masks into a dictionary
       H - height of image 
       W - width of image
    
    Returns: a dictionary containing one prediction mask for each frame with the frame name as key and its predicted mask as value; 
             For each frame, the binary masks of different classes are conbined into a single prediction mask;
             The prediction mask for each frame is a 1024 x 1280 map with each value representing the class id for the pixel;
             
    """    
    endovis_masks = dict()
        
    for seq in binary_masks.keys():
        
        for frame in binary_masks[seq].keys():
            
            endovis_mask = np.zeros((H, W))
    
            binary_masks_list = binary_masks[seq][frame]

            binary_masks_list = sorted(binary_masks_list, key=lambda x: x["mask_quality"])
           
            for binary_mask in binary_masks_list:
                mask_name  = binary_mask["mask_name"]

                if 'class' in mask_name:

                    full_mask_name= mask_name.split('_class')[0]

                    predicted_label = int(re.search(r"class(\d+)", mask_name).group(1))

                    if predicted_label == 0:
                        predicted_label = num_classes 

                    mask = binary_mask["mask"].numpy()
                    endovis_mask[mask==mask.max()] = predicted_label
                    

            endovis_mask = endovis_mask.astype(int)

            endovis_masks[full_mask_name] = endovis_mask
    
    return endovis_masks, endovis_mask


def eval_hysterectomy(endovis_masks, gt_endovis_masks, num_classes=2):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """
    endovis_results = dict()
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    for file_name, prediction in endovis_masks.items():

        full_mask = gt_endovis_masks[file_name]
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:

            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes + 1): 

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(9)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
    endovis_results["IoU"] = round(mean_im_iou*100,3)
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3)
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results


def compute_mask_IU_endovis(masks, target):
    """compute iou used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def read_gt_hysterectomy_mask(data_root_dir, mode, version, num_classes):
    gt_endovis_masks={}
    if mode == "train":
        list_videos = ['Hyst_BB_1.20.23b', 'Hyst_SurgU_3.21.23a', 'AutoLaparo']
    
    elif mode == "val":
        list_videos = ['Hyst_JS_1.30.23','Hyst_BB_4.14.23']

    for name in list_videos: ## video name
        if os.path.isdir(os.path.join(data_root_dir, name)):
            vid_dir = os.path.join(data_root_dir, name)

            for frame_n in os.listdir(vid_dir):

                mask_dir = os.path.join(vid_dir, frame_n, str(version),'binary_annotations')

                if os.path.isdir(mask_dir):
                    annotations = np.zeros((256, 256))
                    for f in os.listdir(mask_dir):
                        mask_name = osp.join(mask_dir, f)
                        base, ext = os.path.splitext(mask_name)
                        full_mask_name=mask_name.split(f"_class")[0]

                        if 'class' in f:

                            predicted_label = int(re.search(r"class(\d+)", f).group(1))
                            if predicted_label == 0:
                                predicted_label = num_classes

                            mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
                            mask = cv2.resize(mask, (256,256))
                            annotations[mask==mask.max()] = predicted_label
                            # annotations[mask==mask.max()] = 1
                    gt_endovis_masks[full_mask_name] = torch.from_numpy(annotations)

    return gt_endovis_masks, annotations


def create_endovis_masks(binary_masks, H, W):
    """given the dictionary containing all predicted binary masks, compute final prediction of each frame and organise the prediction masks into a dictionary
       H - height of image 
       W - width of image
    
    Returns: a dictionary containing one prediction mask for each frame with the frame name as key and its predicted mask as value; 
             For each frame, the binary masks of different classes are conbined into a single prediction mask;
             The prediction mask for each frame is a 1024 x 1280 map with each value representing the class id for the pixel;
             
    """
    
    endovis_masks = dict()
    
    for seq in binary_masks.keys():
        
        for frame in binary_masks[seq].keys():
            
            endovis_mask = np.zeros((H, W))
    
            binary_masks_list = binary_masks[seq][frame]

            binary_masks_list = sorted(binary_masks_list, key=lambda x: x["mask_quality"])
           
            for binary_mask in binary_masks_list:
                mask_name  = binary_mask["mask_name"]
                mask_key =  mask_name.split('annotations/')[1].split('_class')[0]
                predicted_label = int(re.search(r"class(\d+)", mask_name).group(1))
                predicted_label = 1 
                # predicted_label 
                mask = binary_mask["mask"].numpy()
                endovis_mask[mask==1] = predicted_label

            endovis_mask = endovis_mask.astype(int)

            endovis_masks[f"{mask_key}.png"] = endovis_mask
    
    return endovis_masks


def eval_endovis(endovis_masks, gt_endovis_masks, num_classes):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    endovis_results = dict()
    num_classes = 7
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    for file_name, prediction in endovis_masks.items():
       
        full_mask = gt_endovis_masks[file_name]
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes + 1):

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            pdb.set_trace()
            print(current_pred.max(), current_target.pred())

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(9)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
    endovis_results["IoU"] = round(mean_im_iou*100,3)
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3)
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results


def read_gt_endovis_masks(data_root_dir = "../data/endovis_2018",
                          mode = "val", 
                          fold = None):
    
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_endovis_masks = dict()
    
    if "2018" in data_root_dir:
        gt_endovis_masks_path = osp.join(data_root_dir, mode, "annotations")
        for seq in os.listdir(gt_endovis_masks_path):
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, seq)):
                full_mask_name = f"{seq}/{mask_name}"
                mask = cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE)
                mask = torch.from_numpy(cv2.resize(mask, (256,256)))
                gt_endovis_masks[full_mask_name] = mask
                
    elif "2017" in data_root_dir:
        if fold == "all":
            seqs = [1,2,3,4,5,6,7,8]
            
        elif fold in [0,1,2,3]:
            fold_seq = {0: [1, 3],
                        1: [2, 5],
                        2: [4, 8],
                        3: [6, 7]}
            
            seqs = fold_seq[fold]
        
        gt_endovis_masks_path = osp.join(data_root_dir, "0", "annotations")
        
        for seq in seqs:
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, f"seq{seq}")):
                full_mask_name = f"seq{seq}/{mask_name}"
                mask = cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE)
                mask = torch.from_numpy(cv2.resize(mask, (256,256)))
                gt_endovis_masks[full_mask_name] = mask
            
            
    return gt_endovis_masks


def print_log(str_to_print, log_file):
    """Print a string and meanwhile write it to a log file
    """
    print(str_to_print)
    with open(log_file, "a") as file:
        file.write(str_to_print+"\n")
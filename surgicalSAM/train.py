import sys
sys.path.append("..")
from comet_ml import Experiment, ExistingExperiment
import os
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset, HysterectomyDataset
from segment_anything_Surg import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import *
from model_forward import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses
from tqdm import tqdm
import io
import matplotlib.pyplot as plt
import pdb


def log_image(img, legend, exp, epoch):
    buf = io.BytesIO()
    img.savefig(buf, format='png')
    buf.seek(0)
    exp.log_image(buf,legend, step=epoch)


def log_img_batch(batch_img, legend, exp, epoch):

    batch_size = len(batch_img)
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)

    for idx, img in enumerate(batch_img):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.imshow(img)

    log_image(fig, legend, exp, epoch)
    plt.close(fig)




print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017", "hysterectomy"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
parser.add_argument('--data_root', type=str, default='../../data/', help='path to data directory')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
parser.add_argument('--model', type=str, default=None, help='path to pretrained model or pth to resume training')
parser.add_argument('--batch_size', type=int, default=32)


args = parser.parse_args()

print("======> Set Parameters for Training" )
dataset_name = args.dataset
fold = args.fold
thr = 0.7
seed = 666  
batch_size = args.batch_size
vit_mode = "h"

# set seed for reproducibility 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    num_tokens = 2
    data_root_dir = args.data_root
    val_dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                   mode="val",
                                   vit_mode = "h")
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, mode = "val")
    num_epochs = 500
    lr = 0.001
    save_dir = "./work_dirs/endovis_2018/"

elif "17" in dataset_name:
    num_tokens = 4
    data_root_dir = f"../data/{dataset_name}"
    val_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                   mode = "val",
                                   fold = fold, 
                                   vit_mode = "h",
                                   version = 0)
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", 
                                             fold = fold)
    num_epochs = 2000
    lr = 0.0001
    save_dir = f"./work_dirs/endovis_2017/{fold}"

elif dataset_name == 'hysterectomy':
    num_tokens = 2
    val_dataset = HysterectomyDataset(data_root_dir = args.data_root, mode = "val",
                                      vit_mode = "h", version = 1, num_classes=args.num_classes)
    
    train_dataset = HysterectomyDataset(data_root_dir = args.data_root, mode = "train",
                                      vit_mode = "h", version = 1, num_classes=args.num_classes)
    
    weights = train_dataset.dataset_weights().cuda()
    
    gt_endovis_masks, gt_annotations = read_gt_hysterectomy_mask(data_root_dir = args.data_root, mode="val",
                                                 version=1, num_classes=args.num_classes)
    num_epochs = 10000
    lr = 0.001
    save_dir = "/CMF/data/lumargot/"

    
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("======> Load SAM" )
if vit_mode == "h":
    sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = True

print("======> Load Prototypes and Prototype-based Prompt Encoder" )

learnable_prototypes_model = Learnable_Prototypes(num_classes = args.num_classes, feat_dim = 256).cuda()
protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64,
                                                    num_classes=args.num_classes,
                                                    num_tokens = num_tokens).cuda()

 
with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {k.split("prompt_encoder.point_embeddings.")[-1]: v for k, v in state_dict.items() if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)}
    sam_pn_embeddings_weight_ckp = {"0.weight": torch.concat([sam_pn_embeddings_weight['0.weight'] for _ in range(num_tokens)], dim=0),
                                    "1.weight": torch.concat([sam_pn_embeddings_weight['1.weight'] for _ in range(num_tokens)], dim=0)}

    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp)

if args.model :
    print(f"======> Loading weight from {args.model}")

    # load the weight for prototype-based prompt encoder, mask decoder, and prototypes
    checkpoint = torch.load(args.model)
    protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'], strict=False)
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'], strict=False)
    # learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])

for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = True
    
for name, param in protoype_prompt_encoder.named_parameters():
    if "pn_cls_embeddings" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

print("======> Define Optmiser and Loss")
seg_loss_model = DiceLoss().cuda()
class_loss = torch.nn.CrossEntropyLoss(weight=weights)
# class_loss = torch.nn.BCEWithLogitsLoss(weight=weights)
contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()
optimiser = torch.optim.Adam([
            {'params': learnable_prototypes_model.parameters()},
            {'params': protoype_prompt_encoder.parameters()},
            {'params': sam_decoder.parameters()}
        ], lr = lr, weight_decay = 0.0001)


print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok = True) 
log_file = osp.join(save_dir, "log.txt")
# print_log(str(args), log_file)

exp = Experiment(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                        project_name='SurgicalSam',
                        workspace='luciedle')


print("======> Start Training and Validation" )
best_challenge_iou_val = -100.0
best_loss = 10e5
sig = torch.nn.Sigmoid()

val_steps =0
train_steps = 0
val_steps = 0
for epoch in range(num_epochs): 
    
    # # choose the augmentation version to use for the current epoch 
    if epoch % 2 == 0 :
        version = 1 
    else:
        version = int((epoch % 80 + 1)/2)
    if version == 0:
        version = 1 

    # some data augmentation version are failing --> to do: fix it
    if version == 6:
        version = 5

    if version == 14:
        version = 13    
    
    if "18" in dataset_name:
        train_dataset = Endovis18Dataset(data_root_dir = data_root_dir,
                                         mode="train",
                                         vit_mode = vit_mode,
                                         version = 0)
        
    elif "17" in dataset_name:
        train_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                         mode="train",
                                         fold = fold,
                                         vit_mode = vit_mode,
                                         version = version)
        
    elif dataset_name == 'hysterectomy':
        train_dataset = HysterectomyDataset(data_root_dir = args.data_root, 
                                            mode = "train", vit_mode = "h", 
                                            version = version)
        
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # training 
    protoype_prompt_encoder.train()
    sam_decoder.train()
    learnable_prototypes_model.train()

    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch} ")

    all_acc = []
    for step, batch in progress_bar:
        sam_feats, mask_names, cls_ids, masks, class_embeddings = batch

        protoype_prompt_encoder.train()
        sam_decoder.train()
        learnable_prototypes_model.train()


        sam_feats = sam_feats.cuda()
        cls_ids = cls_ids.cuda()
        masks = masks.cuda()
        class_embeddings = class_embeddings.cuda()

        
        prototypes = learnable_prototypes_model()
        
        preds, _ , cls_probs = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, 
                                          sam_decoder, sam_feats, prototypes, cls_ids,
                                          img_size=train_dataset.img_size)

        # preds_cls = torch.argmax(cls_probs, axis=1)
        # one_hot = torch.nn.functional.one_hot(cls_ids,args.num_classes) 
        # pdb.set_trace()
        cls_loss = class_loss(cls_probs, cls_ids)

        # cls_loss =  class_loss(cls_probs, one_hot.double())
        # preds_cls = sig(cls_probs)

        # train_acc = torch.sum(preds_cls == cls_ids).to(float)
        # train_acc/= preds_cls.shape[0]
        # all_acc.append(train_acc.cpu().numpy())

        contrastive_loss = contrastive_loss_model(prototypes, 
                                                  torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(), 
                                                  ref_emb = class_embeddings, 
                                                  ref_labels = cls_ids)
        
        seg_loss = seg_loss_model(preds, masks/255)
    
        loss = seg_loss + contrastive_loss + cls_loss
   
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        exp.log_metric('train/seg loss', seg_loss.item(), step=train_steps)
        exp.log_metric('train/ctrast loss', contrastive_loss.item(), step=train_steps)
        exp.log_metric('train/class loss', cls_loss.item(), step=train_steps)

        val_steps += step

        if step==1:
            log_img_batch(preds.detach().cpu().numpy(), f'pred mask', exp, epoch)
            log_img_batch(masks.detach().cpu().numpy(), f'GT mask', exp, epoch)

    mean_acc = np.sum(np.array(all_acc))/ len(all_acc)
    print(f"training acc : {mean_acc}")


    loss /= step
    exp.log_metric('train loss', loss.item(), step=epoch)



    # validation 
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()
    all_acc = []

    with torch.no_grad():
        prototypes = learnable_prototypes_model()
        val_loss = 0

        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), ncols=70)
        progress_bar.set_description(f"Validation ")
        for step, batch in progress_bar: 
            sam_feats, mask_names, cls_ids, masks, class_embeddings = batch
            
            sam_feats = sam_feats.cuda()
            cls_ids = cls_ids.cuda()

            one_hot = torch.nn.functional.one_hot(cls_ids,args.num_classes) 

            
            preds, preds_quality, cls_probs = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, 
                                                           sam_decoder, sam_feats, prototypes, cls_ids,
                                                           img_size=val_dataset.img_size)
 
            # preds_cls = sig(cls_probs)

            # val_acc = torch.sum(preds_cls == cls_ids).to(float)
            # val_acc/= preds_cls.shape[0]
            # all_acc.append(val_acc.cpu().numpy())

            binary_masks = create_hyst_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

            # cls_loss =  class_loss(cls_probs, one_hot.double())
            cls_loss = class_loss(cls_probs, cls_ids)
            contrastive_loss = contrastive_loss_model(prototypes, torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(), ref_emb = class_embeddings.cuda(), ref_labels = cls_ids)
            seg_loss = seg_loss_model(preds.cuda(), masks.cuda()/255)
            val_loss += (seg_loss + contrastive_loss + cls_loss)

            exp.log_metric('val/seg loss', seg_loss.item(), step=val_steps)
            exp.log_metric('val/ctrast loss', contrastive_loss.item(), step=val_steps)
            exp.log_metric('val/class loss', cls_loss.item(), step=val_steps)

            
            if step ==1:
                log_img_batch(preds.detach().cpu().numpy(), f'predicted val mask', exp, epoch)
                log_img_batch(masks.detach().cpu().numpy(), f'GT val mask', exp, epoch)

        val_loss /=step
        exp.log_metric('val loss', val_loss.item(), step=epoch)
        # exp.log_metric('val/ctrast loss', contrastive_loss.item(), step=step)

    if val_loss < best_loss :
        print(f"found better loss: old {best_loss}, saving {val_loss}")
        best_loss = val_loss
        torch.save({
            'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
            'sam_decoder_state_dict': sam_decoder.state_dict(),
            'prototypes_state_dict': learnable_prototypes_model.state_dict(),
        }, osp.join(save_dir,f'model_ckp_{val_loss}.pth'))

    if 'endovis' in dataset_name:
        endovis_masks = create_endovis_masks(binary_masks, val_dataset.img_size, val_dataset.img_size)
        endovis_results = eval_endovis(endovis_masks, gt_endovis_masks, num_classes=args.num_classes)
    else:
        endovis_masks, annotations = create_hysterectomy_masks(binary_masks, val_dataset.img_size, val_dataset.img_size, num_classes=args.num_classes)
        endovis_results = eval_hysterectomy(endovis_masks, gt_endovis_masks, num_classes=args.num_classes)

    print_log(f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results} ", log_file)
    mean_acc = np.sum(np.array(all_acc))/ len(all_acc)
    print(f"val acc : {mean_acc}")
    
    if endovis_results["challengIoU"] > best_challenge_iou_val:
        best_challenge_iou_val = endovis_results["challengIoU"]
        
        torch.save({
            'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
            'sam_decoder_state_dict': sam_decoder.state_dict(),
            'prototypes_state_dict': learnable_prototypes_model.state_dict(),
        }, osp.join(save_dir,'model_ckp.pth'))

        print_log(f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)
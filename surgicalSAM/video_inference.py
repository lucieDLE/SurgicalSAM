import sys
sys.path.append("..")
import os
import argparse
import torch 
import numpy as np 
import cv2 
import pdb

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from einops import rearrange

from model import Prototype_Prompt_Encoder, Learnable_Prototypes
from segment_anything_Surg import sam_model_registry, SamPredictor


class TestDataset(Dataset):
    def __init__(self, data_root_dir = "../hysterectomy/Clips/", 
                 mode = "val", 
                 img_size=256,
                 num_classes=1):
        
        self.img_size = img_size
        self.num_classes = num_classes
       
        # directory containing all binary annotations

        if mode == "train":
            list_videos = ['Hyst_BB_1.20.23b', 'Hyst_SurgU_3.21.23a', 'AutoLaparo']
        
        elif mode == "val":
            list_videos = ['Hyst_JS_1.30.23','Hyst_BB_4.14.23']

        elif mode =='test':
            list_videos = ['Hyst_SurgU_3.21.23b']

        self.img_list=[]
        for name in list_videos: ## video name
            if os.path.isdir(os.path.join(data_root_dir, name)):
                vid_dir = os.path.join(data_root_dir, name)

                for frame_n in os.listdir(vid_dir):

                    img_dir = os.path.join(vid_dir, frame_n,'0/images')

                    if os.path.isdir(img_dir):
                        for f in os.listdir(img_dir):
                            if os.path.splitext(f)[1] == '.png':

                                file_version = os.path.join(img_dir, f)
                                self.img_list.append(file_version)

        self.img_list = self.img_list[:30]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        
        original_frame = cv2.imread(img_path)

        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.resize(original_frame, (self.img_size,self.img_size))

            
        return original_frame
 

def overlay(image, pred, thr=0.8):
    pred = pred[0].cpu().numpy()
    image = image.cpu().numpy().astype(int)

    norm_thr = 1

    norm_pred = pred * 255
    # print(norm_pred.min(), norm_pred.max(), norm_thr)

    mask = np.zeros_like(norm_pred)
    mask[norm_pred>norm_thr] = 1
    mask_bool = mask.astype(bool)

    image[mask_bool] = [20, 20, 200]  # Set blue channel to 255

    return image


def main(args):
    print("======> Set Parameters for Inference" )
    dataset_name = args.dataset
    thr = args.thr


    print("======> Load Dataset-Specific Parameters" )
    if dataset_name == 'hysterectomy':
        dataset = TestDataset(data_root_dir = args.data_root, mode = "test",
                                        num_classes=args.num_classes)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


    print("======> Load SAM" )
    sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h" ## we need the encoder 


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).cuda()
    predictor = SamPredictor(sam)
    sam_prompt_encoder = sam.prompt_encoder.cuda()
    sam_decoder = sam.mask_decoder.cuda()


    print("======> Load Prototypes and Prototype-based Prompt Encoder" )
    # define the models
    learnable_prototypes_model = Learnable_Prototypes(num_classes = args.num_classes, feat_dim = 256).cuda()
    prototype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                        hidden_dim_dense = 128, 
                                                        hidden_dim_sparse = 128, 
                                                        size = 64, 
                                                        num_classes=args.num_classes,
                                                        num_tokens = 2).cuda()
                
    # load the weight for prototype-based prompt encoder, mask decoder, and prototypes
    checkpoint = torch.load(args.model)
    prototype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
    learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])

    # set requires_grad to False to the whole model 
    for name, param in sam_prompt_encoder.named_parameters():
        param.requires_grad = False
    for name, param in sam_decoder.named_parameters():
        param.requires_grad = False
    for name, param in prototype_prompt_encoder.named_parameters():
        param.requires_grad = False
    for name, param in learnable_prototypes_model.named_parameters():
        param.requires_grad = False

    out_vid_path = "outfile.mp4"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_vid_path, fourcc, 1.0, (256,256))


    print("======> Start Inference")
    prototype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()

    with torch.no_grad():
        prototypes = learnable_prototypes_model()

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=110)
        for step, batch in progress_bar:
            image = batch[0]

            # 1. predict masks with SAM encoder        
            preds, _ = prediction_sam_surgsam(prototype_prompt_encoder, sam_prompt_encoder, sam_decoder, 
                                              predictor, prototypes, image)
            
            # 2. Construct non-binary masks (multiple class per frame)
            # how ? 


            # 3. prepare image + mask overlay
            img = overlay(image, preds, thr)
            img = img.astype(np.uint8)

            # 4. writing image 
            out.write(img)

    out.release()
    # cv2.destroyAllWindows()
    # cv2.destroyAllWindows()


def prediction_sam_surgsam(prototype_prompt_encoder, sam_prompt_encoder, sam_decoder, 
                           predictor,prototypes, images, cls_ids=None):
        
    ## 1. predict features with SAM encoder
    img_pil_format = images.cpu().numpy()
    img_pil_format=img_pil_format.astype('uint8')

    predictor.set_image(img_pil_format)

    sam_feats = predictor.features.squeeze().permute(1, 2, 0)

    sam_feats = sam_feats.unsqueeze(dim=0)
    sam_feats = rearrange(sam_feats, 'b h w c -> b (h w) c')
    
    ## T remove to test if working or not
    # cls_ids = torch.tensor([1]).cuda()

    # 2. compute embedding with class_id --> check output size + meaning 
    if cls_ids != None:
        dense_embeddings, sparse_embeddings = prototype_prompt_encoder(sam_feats, prototypes, cls_ids)
    else:
        dense_embeddings, sparse_embeddings, top_idx, top_val  = prototype_prompt_encoder.prediction_without_prompt(sam_feats, prototypes)

    pred = []
    pred_quality = []
    sam_feats = rearrange(sam_feats,'b (h w) c -> b c h w', h=64, w=64)
 

    for class_idx in range(dense_embeddings.shape[1]):
        dense_embedding_per_class = dense_embeddings[:,class_idx,:,:,:]
        for dense_embedding, sparse_embedding, features_per_image in zip(dense_embedding_per_class.unsqueeze(1), sparse_embeddings.unsqueeze(1), sam_feats):    
            low_res_masks_per_image, mask_quality_per_image = sam_decoder(
                    image_embeddings=features_per_image.unsqueeze(0), ## similar values
                    image_pe=sam_prompt_encoder.get_dense_pe(), ## same
                    sparse_prompt_embeddings=sparse_embedding,
                    dense_prompt_embeddings=dense_embedding, 
                    multimask_output=False,
                )
            
            pred.append(low_res_masks_per_image)
            pred_quality.append(mask_quality_per_image.detach().cpu())
        
    pred = torch.cat(pred,dim=0).squeeze(1)

    pred_quality = torch.cat(pred_quality,dim=0)

    return pred, pred_quality


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Tool segmentation and tracking Inference')
    parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017", "hysterectomy"], help='specify dataset')
    parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
    parser.add_argument('--data_root', type=str, default='../../data/', help='path to data directory')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--model', type=str, default=None, help='path to surgical sam model')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--outfile', type=str, default='./out.mp4')
    parser.add_argument('--thr', type=int, default=0.8)


    args = parser.parse_args()

    main(args)
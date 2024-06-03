import torch 
import torch.nn as nn 
from einops import rearrange

import pdb

class Prototype_Prompt_Encoder(nn.Module):
    def __init__(self, feat_dim=256, 
                        hidden_dim_dense=128, 
                        hidden_dim_sparse=128, 
                        size=64, 
                        num_tokens=8,
                        num_classes=1):
                
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        
        ## Adding classification head to find tool category
        self.classification_head = nn.Linear(feat_dim*num_classes*2, num_classes)
       
        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
        
        
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)] # one for positive and one for negative 

        self.num_classes= num_classes
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
                
    def forward(self, feat, prototypes, cls_ids):
  
        cls_prompts = prototypes.unsqueeze(-1)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)
        
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)

        # compute similarity matrix 
        sim = torch.matmul(feat, cls_prompts)
        
        # compute class-activated feature
        feat =  feat + feat*sim

        feat_sparse = feat.clone()
        
        # compute dense embeddings
        one_hot = torch.nn.functional.one_hot(cls_ids,self.num_classes) 
        feat = feat[one_hot ==1]
        feat = rearrange(feat,'b (h w) c -> b c h w', h=64, w=64)
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))


        # compute classification probabilities

        bs, feat_dim, size, size  = dense_embeddings.shape

        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=self.num_classes)
        
        flatten_sparse_embeddings = sparse_embeddings.reshape((bs, -1))
        cls_probs = self.classification_head(flatten_sparse_embeddings)
            

        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)
        

        sparse_embeddings = sparse_embeddings + pos_embed + neg_embed
        
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
        return dense_embeddings, sparse_embeddings, cls_probs
    


    def prediction_without_prompt(self, feat, prototypes):
        cls_prompts = prototypes.unsqueeze(-1)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)
        
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)

        # compute similarity matrix 
        sim = torch.matmul(feat, cls_prompts)
        feat =  feat + feat*sim

        feat_sparse = feat.clone()

        feat_reshaped = rearrange(feat, 'b num_cls (h w) c -> b num_cls h w c', h=64, w=64)
        batch_size, num_classes, h, w, embedding_dim = feat_reshaped.shape


        ## need to double / triple check that 
        top_val, top_idx = sim.topk(num_classes, dim=1)  # k is the number of instruments we can have per images

        mask = torch.zeros(batch_size, num_classes, h * w, dtype=torch.bool, device=feat.device)

        for batch_idx in range(batch_size):
            # class_indices = top_idx[batch_idx]
            class_indices = top_idx[batch_idx, :, :, 0]  # Shape: (k, height * width)
            for class_idx in range(num_classes):
                mask[batch_idx, class_idx] = (class_indices == class_idx).any(dim=0)

        mask = mask.reshape(batch_size, num_classes, h * w).unsqueeze(-1)

        selected_features = feat_reshaped.reshape(batch_size, num_classes, h * w, embedding_dim)
        selected_features = torch.masked_select(selected_features, mask)

        # --------
        selected_features = selected_features.view(-1, embedding_dim, h, w)

        # Compute dense embeddings for selected features
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(selected_features)))
        dense_embeddings = rearrange(dense_embeddings, '(b c) d h w -> b c d h w', b=batch_size, c=9)
        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=self.num_classes)
                    
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
        return dense_embeddings, sparse_embeddings, top_idx, top_val



class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=7 , feat_dim=256):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        
    def forward(self):
        return self.class_embeddings.weight
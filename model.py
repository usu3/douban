import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn
from transformers import BertTokenizer,BertModel
import accelerate

class mymodel(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=128)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=256)
        self.lm=BertModel.from_pretrained("bert-base-chinese")
        self.norm1=nn.BatchNorm1d(256)
        self.norm2=nn.BatchNorm1d(128)
        #self.norm1=nn.LayerNorm(256)
        #self.norm2=nn.LayerNorm(128)
        self.act=nn.ReLU()
        self.softmax=nn.Softmax(dim=0)
        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=384, out_features=128)
        self.output = nn.Linear(in_features=131, out_features=1)

        
    def forward(self, user_id, item_id,user_follow,feature_item,feature_tags):
        
        # Pass through embedding layers
        user_embedded = self.user_interaction_embde(user_id,user_follow)#dim:(batch,128)
        item_embedded = self.item_embedding(item_id)#dim:(batch,256)
        tokenized_tags=self.tokenizer(feature_tags,return_tensors='pt',padding=True,truncation=True,max_length=200).to(user_embedded.device)        
        tag_embdedded=self.lm(**tokenized_tags)['pooler_output'].reshape(-1,768)#dim:(batch,768)
        # Concat the two embedding layers
        item_vector = self.act(self.norm1(self.fc1(torch.cat([tag_embdedded, item_embedded], dim=1))+item_embedded))#dim:(batch,256)
        item_user_vector=self.act(self.norm2(self.fc2(torch.cat([user_embedded, item_vector], dim=1))+user_embedded))#dim:(batch,128)
        logits=self.output(torch.cat([item_user_vector,feature_item],dim=1))#dim:(batch,1)
        return logits
    def user_interaction_embde(self,user_id,user_follow):
        lyst=[]
        for i in range(len(user_follow)):
            if len(user_follow[i])>1:
                follow_embedded=self.user_embedding(user_follow[i])#nx128
                user_embedded=self.user_embedding(user_id[i])#1x128
                user_embedded=user_embedded.reshape(-1,128)
                user_follow_embedded=torch.cat([user_embedded,follow_embedded],dim=0)#(n+1)*128
                atten_weights=self.softmax(torch.matmul(user_follow_embedded,user_embedded.T))#(n+1)*1
                user_embedding=torch.matmul(atten_weights.T , user_follow_embedded)  
            else:
                user_embedding=self.user_embedding(user_id[i])
                user_embedding=user_embedding.reshape(-1,128)
            lyst.append(user_embedding)
        return torch.cat(lyst,0)#(batch,128)

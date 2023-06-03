from datasets import load_dataset,load_from_disk
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import BertTokenizer
class douban_set(Dataset):
    def __init__(self,path):
        super().__init__()
        self.data=self.construct_dataset(path)
        #self.tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
    def construct_dataset(self,path):
        def process(x):
            x['item_rate']=(float(x['item_rate']) if x['item_rate']!='None' else 0.)
            x['rating']=(1. if x['rating']>=3 else 0.)
            return x
        data=load_from_disk(path)
        data=data.map(process)
        return data
    def __getitem__(self, index):
        features_tags='[SEP]'.join(self.data[index]['item_tags'])
        user_id=self.data[index]['user_index']
        user_follow=self.data[index]['user_follow']
        item_id=self.data[index]['item_index']
        features_item=[self.data[index]['item_reviews_count'],self.data[index]['item_rate'],self.data[index]['item_rate_people']]
        label=self.data[index]['rating']
        return torch.tensor(user_id),torch.tensor(item_id),torch.tensor(user_follow),torch.tensor(features_item),features_tags,torch.tensor(label)
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    user_id=torch.stack([val[0] for val in batch])
    item_id=torch.stack([val[1] for val in batch])
    user_follow=[val[2] for val in batch]
    features_item=torch.stack([val[3] for val in batch])
    features_tags=[val[4] for val in batch]
    label=torch.stack([val[5] for val in batch])
    datum={'features':{'user_id':user_id,'item_id':item_id,'user_follow':user_follow,'feature_item':features_item,'feature_tags':features_tags},'label':label}
    return datum       
from accelerate import Accelerator
from tqdm import tqdm
from torch import nn
import os
import time
class trainer(object):
    def __init__(self,model,optimizer,train_loader,val_loader,log_dir,save_dir):
        self.accelerator=Accelerator(log_with='tensorboard',project_dir=log_dir)
        self.accelerator.init_trackers(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        self.model,self.optimizer,self.train_loader=self.accelerator.prepare(model,optimizer,train_loader)
        self.val_loader=self.accelerator.prepare(val_loader)
        self.steps = 0
        self.save_dir=save_dir
    
    def train(self,epochs,loss_fn):
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for batch in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                inputs, targets = batch['features'],batch['label']
                outputs = self.model(**inputs)
                loss = loss_fn(outputs, targets.reshape(-1,1).float())
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.steps += 1
                self.accelerator.log({"train_loss": loss.item()}, step=self.steps)
                if self.steps % 30000 == 0:
                    accuracy, precision, recall, f1, val_loss = self.eval()
                    self.accelerator.log({"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Val_Loss": val_loss}, step=self.steps)
                    self.accelerator.wait_for_everyone()
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.save_dir,f'model_{self.steps}.pth'))
                    #print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, Val Loss: {val_loss}")
        self.accelerator.end_training()
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            val_loss=0
            i=0
            for batch in tqdm(self.val_loader):
                inputs, targets = batch['features'],batch['label']
                outputs = self.model(**inputs)
                tp += ((outputs > 0.5) & (targets == 1)).sum().item()
                tn += ((outputs < 0.5) & (targets == 0)).sum().item()
                fp += ((outputs > 0.5) & (targets == 0)).sum().item()
                fn += ((outputs < 0.5) & (targets == 1)).sum().item()
                val_loss+= nn.BCEWithLogitsLoss()(outputs, targets.reshape(-1,1).float()).item()
                i+=1
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            val_loss=val_loss/i
        self.model.train()
        return accuracy, precision, recall, f1, val_loss
if __name__=='__main__':
    import os
    import torch
    from model import mymodel
    import doubanset
    from torch.utils.data import DataLoader,Dataset
    from data_process import construct_movieindex,construct_userindex
    import pandas as pd
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    df_user=pd.read_csv('user.csv',index_col=0)
    df_item=pd.read_csv('movie.csv',index_col=0)
    df_userfollow=pd.read_csv('user_following.csv')
    df_userfollow.dropna(inplace=True)
    user_index_map=construct_userindex(df_user,df_userfollow)
    item_index_map=construct_movieindex(df_item)

    train_set=doubanset.douban_set('train_sample')
    val_set=doubanset.douban_set('test_sample')
    train_loader=DataLoader(train_set,70,shuffle=True,collate_fn=doubanset.collate_fn)
    val_loader=DataLoader(val_set,70,shuffle=True,collate_fn=doubanset.collate_fn)
    model=mymodel(len(user_index_map)+1,len(item_index_map)+1)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    #loss=
    t=trainer(model,optimizer,train_loader,val_loader,'log_dir','save_dir')
    t.train(10,torch.nn.BCEWithLogitsLoss())


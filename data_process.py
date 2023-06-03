import pandas as pd
import json
from datasets import load_dataset,load_from_disk
import datasets
import random
from tqdm import tqdm
def construct_userindex(df,df_userfollow):
    user_index_map={}
    val=1
    for key in df['name'].unique():
        user_index_map[key]=val
        val+=1
    for i in range(len(df_userfollow)):
        tmp=df_userfollow.iloc[i,1].strip('[]').split(',')
        for x in tmp:
            if x.strip("' '") not in user_index_map:
                user_index_map[x.strip("' '")]=val
                val+=1
    return user_index_map#{int:int 1-inf}
def construct_movieindex(df):
    item_index_map={}
    val=1
    for key in df['id'].unique():
        item_index_map[key]=val
        val+=1
    return item_index_map#{int:int 1-inf}

def write_user_item_json(path):
    lyst=[]
    #f1=open(path, 'w')
    for i in tqdm(range(len(df_user))):
        user_name=df_user.iloc[i,1]
        user_index=user_index_map[user_name]
        tmp_user_follow=df_userfollow.loc[df_userfollow['user_id']==user_name,'following_list'].iloc[0].strip('[]').split(',')
        user_follow=[]
        for x in tmp_user_follow:
            user_follow.append(user_index_map[x.strip("' '")])
            
        for interaction in df_user.iloc[i,2].strip('{}').split(','):
            item,rate=interaction.split(':')
            try:
                item_index,rate=item_index_map[int(item.strip(" u''"))],int(rate.strip(" u''"))
            except KeyError:
                continue
            
            item_rate=df_item.loc[df_item['id']==int(item.strip(" u''")),'rate'].iloc[0]#float
            try:
                item_rat_people=int(df_item.loc[df_item['id']==int(item.strip(" u''")),'rating_people'].iloc[0].strip('人评价'))#int
            except ValueError:
                item_rat_people=0
            try:
                item_reviews_count=int(df_item.loc[df_item['id']==int(item.strip(" u''")),'reviews_count'].iloc[0].strip('全部条'))#int
            except ValueError:
                item_reviews_count=0
            item_tags=(df_item.loc[df_item['id']==int(item.strip(" u''")),'type'].iloc[0].split('/') 
                       if df_item.loc[df_item['id']==int(item.strip(" u''")),'tags'].iloc[0].split('#')==None 
                       else df_item.loc[df_item['id']==int(item.strip(" u''")),'tags'].iloc[0].split('#'))#list
            
            persample={'user_name':user_name,'user_index':user_index,'user_follow':user_follow,
                       'item_name':int(item.strip(" u''")),'item_index':item_index,
                       'rating':rate,#below is item features
                       'item_rate':item_rate,
                       'item_rate_people':item_rat_people,
                       'item_reviews_count':item_reviews_count,
                       'item_tags':item_tags}
            lyst.append(persample)
    with open(path,'w',encoding='utf-8') as f1:
        f1.write(json.dumps(lyst))

def candidate():    
    postivelist={}
    for i in range(len(df_user)):
        tmp=df_user.iloc[i,2].strip('{}').split(',')
        postivelist[df_user.iloc[i,1]]=list(map(lambda x: int(x.split(':')[0].strip(" u''")),tmp))
    return postivelist

def generate_negetative_sample(path,trainset,num_negatives):
    train_negetative=[]
    for i in tqdm(range(len(trainset))):
        user_name=trainset[i]['user_name']
        user_index=trainset[i]['user_index']
        user_follow=trainset[i]['user_follow']
        for _ in range(num_negatives):
            negative_item=random.choice(list(item_index_map.keys()))
            while negative_item in postivelist[user_name]:
                negative_item=random.choice(list(item_index_map.keys()))
            #item_feature 
            item_index=item_index_map[negative_item]   
            item_rate=df_item.loc[df_item['id']==negative_item,'rate'].iloc[0]
            try:
                item_rat_people=int(df_item.loc[df_item['id']==negative_item,'rating_people'].iloc[0].strip('人评价'))#int
            except ValueError:
                item_rat_people=0
            try:
                item_reviews_count=int(df_item.loc[df_item['id']==negative_item,'reviews_count'].iloc[0].strip('全部条'))#int
            except ValueError:
                item_reviews_count=0
            item_tags=(df_item.loc[df_item['id']==negative_item,'type'].iloc[0].split('/') 
                       if df_item.loc[df_item['id']==negative_item,'tags'].iloc[0].split('#')==None 
                       else df_item.loc[df_item['id']==negative_item,'tags'].iloc[0].split('#'))#list
            persample={'user_name':user_name,'user_index':int(user_index),'user_follow':user_follow,
                       'item_name':int(negative_item),'item_index':item_index,
                       'rating':0,  #below is item features
                       'item_rate':item_rate,
                       'item_rate_people':item_rat_people,
                       'item_reviews_count':item_reviews_count,
                       'item_tags':item_tags}
            train_negetative.append(persample)
    with open(path,'w',encoding='utf-8') as f1:
        f1.write(json.dumps(train_negetative))
                   
    

    
if __name__=='__main__':
    df_user=pd.read_csv('user.csv',index_col=0)
    df_item=pd.read_csv('movie.csv',index_col=0)
    df_userfollow=pd.read_csv('user_following.csv')
    df_userfollow.dropna(inplace=True)
    user_index_map=construct_userindex(df_user,df_userfollow)
    item_index_map=construct_movieindex(df_item)
    write_user_item_json('douban_user_item.json')
    postivelist=candidate()
    ##train_test_split
    data1=load_dataset('json',data_files='douban_user_item.json',cache_dir='D:\\docc\\fdu\\recommend_system\\.cache')
    train_set=data1['train'].shuffle(seed=42)
    split_set=train_set.train_test_split(test_size=0.2)
    split_set.save_to_disk('./train_test_split')
    disk_datasets = load_from_disk("./train_test_split")
    generate_negetative_sample('douban_negetative_user_item.json',disk_datasets['train'],2)
    #concate
    dd=load_dataset('json',data_files='douban_negetative_user_item.json')
    all_train=datasets.concatenate_datasets([disk_datasets['train'],dd['train']])
    all_train=all_train.shuffle(seed=42)
    all_train.save_to_disk('./train_sample')
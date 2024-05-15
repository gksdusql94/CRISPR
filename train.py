#%%
from datetime import datetime,timedelta
import sys,os,copy,ast,socket,random,math,webbrowser,getpass,time,shutil
import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt

import torch,os,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# %%

#%%
def LtN(DNA):
    DNA=DNA.lower()
    mapping = {'a': 0, 'c': 1, 't': 2, 'g': 3,'x':4,'d':4}
    encoded = torch.zeros((len(DNA), 5), dtype=torch.long)
    for i, nucleotide in enumerate(DNA):
        encoded[i, mapping[nucleotide]] = 1
    return encoded

def NtS(del_target):
    string1=''
    for item in del_target.squeeze():
        string1=string1+str(item.item())
    return string1


from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Revised SequenceEmbedding
class SequenceEmbedding(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super(SequenceEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
    def forward(self, x):
        # For each row, find all active indices and sum their embeddings
        # as you can see it is using like looping each row with one 1's and embed, and then add each one of them together i think?
        embeddings = []
        for row in x:
            active_indices = row.nonzero(as_tuple=True)[0]
            embedded = self.embedding(active_indices)
            embeddings.append(embedded.sum(dim=0))
            
        return torch.stack(embeddings)

# Embedding for scalar data
class ScalarEmbedding(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(ScalarEmbedding, self).__init__()
        self.fc = nn.Linear(num_features, embedding_dim)
        
    def forward(self, x):
        return self.fc(x)

class Feedforward(nn.Module):
    def __init__(self,n_emb):
        super().__init__()
        self.feedforward=nn.Sequential(

            nn.Linear(n_emb,4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb,n_emb), # this is projection layer
            nn.Dropout(DROPOUT)
        )

    def forward(self,x):
        x=self.feedforward(x)
        return x
# y_hat=model(x4)
class Head(nn.Module):
    def __init__(self,heads):
        super().__init__()    
        # self.linear1=nn.Linear(LENGTH*CHARN,out1*2,bias=False)
        # self.linear2=nn.Linear(out1*2,out1,bias=False)
        self.key_linear=nn.Linear(N_EMB,heads,bias=False)
        self.query_linear=nn.Linear(N_EMB,heads,bias=False)
        self.value_linear=nn.Linear(N_EMB,heads,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(LENGTH,LENGTH)))
        self.dropout=nn.Dropout(DROPOUT)
    def forward(self,x):
        B,T,C=x.shape
        xk=self.key_linear(x) # B,T,H
        # print("\n>> xk.shape= ", xk.shape)

        xq=self.query_linear(x) # B,T,H
        xv=self.value_linear(x) # B,T,H
        
        WW=xq @ xk.transpose(-2,-1) * C**-0.5
        WW=WW.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        WW=F.softmax(WW,dim=-1) # B,T,T
        # print("\n>> WW.shape= ", WW.shape)
        WW=self.dropout(WW)
        y_hat=WW@xv # (B,T,T @ B,T,H) > B,T,H
        # print("\n>> y_hat.shape= ", y_hat.shape)

        return y_hat
class Multihead(nn.Module):
    def __init__(self,headn,heads):
        super().__init__()
        self.head_list=nn.ModuleList([Head(heads) for _ in range(headn)])
        self.projection=nn.Linear(headn*heads,N_EMB)
        self.dropout=nn.Dropout(DROPOUT)
    def forward(self,x):
        x=torch.cat([head.forward(x) for head in self.head_list],dim=-1)
        x=self.projection(x)
        y_hat=self.dropout(x)
        return y_hat

class Block(nn.Module):
    def __init__(self,n_emb,headn):
        super().__init__()
        headsize=n_emb//headn
        self.multihead=Multihead(HEADN,headsize)
        self.feedforward=Feedforward(n_emb)
        self.layer_norm1=nn.LayerNorm(n_emb)
        self.layer_norm2=nn.LayerNorm(n_emb)

    def forward(self,x):
        # print('hi')
        # print(x)

        x= x + self.multihead(self.layer_norm1(x)) ### B,T,C
        x= x + self.feedforward(self.layer_norm2(x))
        return x

class Model(nn.Module):
    def __init__(self,x_check):
        super().__init__()
        self.x_check1=x_check.shape[1]
        self.EMB=nn.Embedding(CHARN, N_EMB)
        self.EMB2=nn.Embedding(LENGTH,N_EMB)
        self.EMB3=nn.Linear(FEATURE_N,max_length)
        self.blocks=nn.Sequential(
            *[Block(N_EMB,HEADN) for _ in range(BLOCKN)]
        )

        self.layer_norm=nn.LayerNorm(N_EMB)
        # language_modeling_head=nn.Linear(N_EMB,CHARN)
        self.final_layer=nn.Linear(LENGTH*N_EMB,1)
        # linear1=nn.Linear(LENGTH*CHARN,out1*2,bias=False)
        # linear2=nn.Linear(out1*2,out1,bias=False)


    def forward(self,x):
        # B,T=x.shape
        x_check=x[:,:self.x_check1].long()
        x_check

        x_features=x[:,self.x_check1:]
        x_features


        x1=self.EMB(x_check) ### B,T,C


        # x1=embed1(x)
        x2=self.EMB2(torch.arange(LENGTH).to(DEVICE)) ### T,C
        x_features=self.EMB3(x_features)

        x_features=x_features.unsqueeze(dim=-1)


        x1=x1+x_features

        x=x1+x2 ### B,T,C



        x=self.blocks(x)
        x=self.layer_norm(x)
        # print("\n>> x.shape= ", x.shape)
        # time.sleep(5)
        x = x.view(x.size(0), -1)

        x=self.final_layer(x)
        return x

#%%Y:\1_code\1_version\8_attention_from_scratch\ng-video-lecture-master\t3_crispr_ml\win3_2.391_CUDA0_20230820142253.pth
df=pd.read_csv('data/DeepPrime_dataset_final_Feat8.csv')
df
# %%
# df=reset_index(df)
# df

max_edited_length=10

max_length=len(df['WT74_On'][0].lower())+max_edited_length
#%%
max1=500
min1=-1000
df[['Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD']]=(df[['Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD']]-min1)/(max1-min1)
df[['fGCcont1', 'fGCcont2', 'fGCcont3']]=df[['fGCcont1', 'fGCcont2', 'fGCcont3']]/100
max1=20
min1=-100
df[['MFE1','MFE2','MFE3', 'MFE4','MFE5']]=(df[['MFE1','MFE2','MFE3', 'MFE4','MFE5']]-min1)/(max1-min1)


# %%
# p1 seed

random_seed = 1

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)


# p2 make dataloader
batch_size = 64


checkpoint = torch.load('data/data_checkpoint2.pth')

x_check = checkpoint['x'].long()
y = checkpoint['y'].long()
x_check
#%%

x_check=torch.argmax(x_check,dim=-1)
x_check
#%%
#%%
x_features=np.array(df[['Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD','fGCcont1', 'fGCcont2', 'fGCcont3','MFE1','MFE2','MFE3', 'MFE4','MFE5']])
x_features
#%%
x_features = torch.tensor(x_features, dtype=torch.float32)
x_features
#%%

#%%
x_combined=torch.cat([x_check,x_features],dim=1)
print("\n>> x_combined.shape= ", x_combined.shape)

#%%
# Assuming x and y are your data tensors
dataset = MyDataset(x_combined, y)


# Define the split sizes. For example, 80% train and 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#%%

# p3 make model#%

max_edited_length=10
max_length=max_edited_length+74
node1=30
N_EMB=int(30*node1)
N_EMB2=int(3*node1)
LENGTH=max_length
# DEVICE='cuda'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

CHARN=x_check.shape[-1]
FEATURE_N=x_features.shape[-1]
BLOCKN=int(6*1)
HEADS=int(30*node1)
HEADN=int(6*1)
DROPOUT=0.2
model=Model(x_check).to(DEVICE)
EPOCHS=1000
LR=1e-4
optimizer=torch.optim.AdamW(params=model.parameters(),lr=LR)


#%%
# p5 test
ef test_acc(test_loader, model, DEVICE):
    total_mse = 0
    pearson_corr = []
    spearman_corr = []
    count = 0
    with torch.no_grad():
        for data in test_loader:
            x, y_true = data
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)
            y_hat = model(x).squeeze()

            # y_hat을 1차원 배열로 변환
            if y_hat.dim() > 1:
                y_hat = y_hat.squeeze()
            if y_true.dim() > 1:
                y_true = y_true.squeeze()

            mse = ((y_hat - y_true) ** 2).mean().item()
            total_mse += mse * len(x)
            count += len(x)

            try:
                if len(y_hat) == len(y_true): 
                    pearson_corr.append(pearsonr(y_hat.cpu().numpy(), y_true.cpu().numpy())[0])
                    spearman_corr.append(spearmanr(y_hat.cpu().numpy(), y_true.cpu().numpy())[0])
            except Exception as e:
                print(f"Error calculating correlations: {e}")

    average_mse = total_mse / count
    average_pearson = np.mean(pearson_corr) if pearson_corr else 0
    average_spearman = np.mean(spearman_corr) if spearman_corr else 0
    return average_mse, average_pearson, average_spearman
#%%
average_mse,mse1=test_acc(test_loader,model,DEVICE)
mse1

#%%
for i, data in enumerate(train_loader, 0):
    x, y_true = data
    x=x.to(DEVICE)
    y_true=y_true.to(DEVICE)
    y_true = y_true.unsqueeze(1)
    y_true = y_true.float()
    break
y_hat=model(x)
y_hat
#%%
# p4 train
loss_=[]
os.makedirs('g2_logs', exist_ok=True)
txt_filename=f"{socket.gethostname().replace('.','')}_{os.path.basename(__file__)}_{datetime.now().strftime('%Y-%m%d-%H%M%S')}"
filename=f"{txt_filename}.txt"
txt_filename
model_folder=f'g1_models/{txt_filename}/'
os.makedirs(model_folder, exist_ok=True)

criterion_mse = nn.MSELoss()
        


for epoch in range(EPOCHS):

    for i, data in tqdm(enumerate(train_loader, 0)):
        x, y_true = data
        x=x.to(DEVICE)
        y_true=y_true.to(DEVICE)
        y_true = y_true.unsqueeze(1)
        y_true = y_true.float()
        y_hat=model(x)
        # print('here 1,',i,epoch)
        y_hat.shape

        y_hat


        loss=criterion_mse(y_hat,y_true)

        loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_.append(float(loss))
    
    average_mse,mse1=test_acc(test_loader,model,DEVICE)

    print('epoch: ',epoch,loss.item(),average_mse)
    with open('g2_logs/'+filename, 'a+') as file:
        file.write(f"{loss}_{node1}_{batch_size}_{epoch}_{mse1}_{datetime.now().strftime('%Y-%m%d-%H%M%S')},\n")


    if epoch%50==0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        # Save the dictionary to a file

        computer1=socket.gethostname().replace('.','')
        today1=datetime.now().strftime('%Y-%m%d-%H%M%S')
        loss1=round(loss.item(),3)
        rank1=torch.cuda.current_device()
        torch_model_name= f'{computer1}_{today1}_cuda{rank1}_{mse1}_{loss1}.pth'
        
        torch.save(checkpoint, model_folder+torch_model_name)
#%%
average_mse,mse1=test_acc(test_loader,model,DEVICE)
#%%nvidia-smi
average_mse
#%%
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}

# Save the dictionary to a file

computer1=socket.gethostname().replace('.','')
today1=datetime.now().strftime('%Y-%m%d-%H%M%S')
loss1=round(loss.item(),3)
rank1=torch.cuda.current_device()
torch_model_name= f'{computer1}_{today1}_cuda{rank1}_{average_mse}_{loss1}.pth'
model_folder='g1_models/'
os.makedirs(model_folder, exist_ok=True)
torch.save(checkpoint, model_folder+torch_model_name)

# %%

torch.save(model,f'a3_model_v1_{round(average_mse,4)}.pt')
# %%

# def test_acc(test_loader,model,DEVICE):
total_mse=0
count=0
total_loss=0
count1=0
loss_list=[]
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
            x, y_true = data
            x=x.to(DEVICE)
            y_true=y_true.to(DEVICE)
            y_true = y_true.unsqueeze(1)
            y_true = y_true.float()
            y_hat=model(x)
            mse = ((y_hat - y_true) ** 2).mean().item()  # Compute the Mean Squared Error for this batch
            total_mse += mse * len(x)  # Accumulate the total MSE, weighted by batch size
            count += len(x)
            total_loss+=loss.item()
            count1+=1
            loss_list.append(total_mse.item())


average_mse = total_mse / count
# print(f"Test MSE: {average_mse}")
average_mse=round(average_mse,3)
mse1=total_loss/count1
print('mse1:',mse1)
# %%
print('average_mse,',average_mse)
#%%
'pearson'
# %%
plt.plot(loss_list)
plt.show()

# %%

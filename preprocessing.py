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
import getpass,sys,socket
from import_basics import *

df=read_excel('data/DeepPrime_dataset_final_Feat8.csv')
df
# %%
df=reset_index(df)
df
max_edited_length=10

max_length=len(df['WT74_On'][0].lower())+max_edited_length

# %%
# for enu1,item in enumerate(df['type_del']):
#     if item==2:
#         break
for enu1 in range(len(df)):
    column1='type_sub'
    edit_number=2
    if int(df[column1][enu1])==int(1) and int(df['Edit_len'][enu1])==int(edit_number) :
        print("df[column1][enu1]: ",df[column1][enu1])
        break
enu1

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

#%%
i=18935
i=270164
i=enu1+2
total_x=[]
total_y=[]
from tqdm import tqdm
#%%
for i in tqdm(range(len(df))):
    # print("df['type_del'][i]: ",df['type_del'][i])
    DNA=df['WT74_On'][i].lower()
    DNA


    TARGET=df['Edited74_On'][i].lower()
    TARGET










    for ii in range(len(DNA)):
        dna=DNA[ii]
        target=TARGET[ii]
        # print("dna,target: ",dna,target)
        if target != 'x':
            break

    # print("ii: ",ii)



    PBS_start=ii
    PBS_end=ii+df['PBSlen'][i]
    # print("PBS_start: ",PBS_start)
    # print("PBS_end: ",PBS_end)

    RTT_start=PBS_end+1
    RTT_start

    Edit_start=PBS_end+df['Edit_pos'][i]
    Edit_start

    edit_length=df['Edit_len'][i]
    # print("edit_length: ",edit_length)
    edit_pos=df['Edit_pos'][i]
    # print("edit_pos: ",edit_pos)





    if df['type_del'][i]==1:
        type1='deletion'
        TARGET=TARGET[:Edit_start-1]+'d'*edit_length+TARGET[Edit_start-1:]
    elif df['type_ins'][i]==1:
        type1='insertion'
        DNA=DNA[:Edit_start]+'d'*edit_length+DNA[Edit_start:]
    elif df['type_sub'][i]==1:
        type1='substitution'
        # print("DNA[Edit_start-1]: ",DNA[Edit_start-1])
        # print("TARGET[Edit_start-1]: ",TARGET[Edit_start-1])




    DNA


    TARGET

    Edit_end=Edit_start+df['Edit_len'][i]
    Edit_end

    RHA_start=Edit_end

    for iii in reversed(range(min(len(DNA),len(TARGET)))):
        dna=DNA[iii]
        target=TARGET[iii]
        # print("dna,target: ",dna,target)
        if target != 'x':
            break

    RHA_end=iii
    # print("RHA_end: ",RHA_end)



    padding_n=max_edited_length-abs(len(TARGET)-len(DNA))
    padding='x'*padding_n

    len(TARGET)



    # DNA=DNA+padding
    # TARGET=TARGET+padding
    # # %%
    len(DNA)

    # print("\n>> len(TARGET)= ", len(TARGET))


    padding_dna=max_length-len(DNA)
    DNA=DNA+'x'*padding_dna
    DNA

    padding_target=max_length-len(TARGET)
    TARGET=TARGET+'x'*padding_target
    TARGET


    len(TARGET)

    target=LtN(TARGET)

    dna=LtN(DNA)

    dna

    len(dna)



    pbs=torch.zeros(len(TARGET))
    pbs[PBS_start:PBS_end] = 1
    pbs=pbs.long().unsqueeze(dim=1)
    pbs


    del_target=torch.zeros(len(TARGET))


    ins_dna=torch.zeros(len(DNA))


    sub_dna=torch.zeros(len(DNA))




    if df['type_del'][i]==1:
        type1='deletion'
        del_target[Edit_start-1:Edit_end-1]=1
        del_target
        rtt=torch.zeros(len(TARGET))
        rtt[RTT_start-1:RHA_end+1]=1
        rtt[Edit_start-1:Edit_end-1]=0
        rtt=rtt.long().unsqueeze(dim=1)

    elif df['type_ins'][i]==1:
        type1='insertion'
        ins_dna[Edit_start:Edit_end]=1

        rtt=torch.zeros(len(TARGET))
        rtt[RTT_start-1:RHA_end+1]=1
        rtt[Edit_start:Edit_end]=0
        rtt=rtt.long().unsqueeze(dim=1)

    elif df['type_sub'][i]==1:
        type1='substitution'
        sub_dna[Edit_start-1:Edit_end-1]=1

        rtt=torch.zeros(len(TARGET))
        rtt[RTT_start-1:RHA_end+1]=1
        rtt[Edit_start-1:Edit_end-1]=0
        rtt=rtt.long().unsqueeze(dim=1)



    del_target=del_target.long().unsqueeze(dim=1)
    ins_dna=ins_dna.long().unsqueeze(dim=1)
    sub_dna=sub_dna.long().unsqueeze(dim=1)

    # print("\n>> target.shape= ", target.shape)
    # print("\n>> dna.shape= ", dna.shape)
    # print("\n>> pbs.shape= ", pbs.shape)
    # print("\n>> rtt.shape= ", rtt.shape)
    # print("\n>> del_target.shape= ", del_target.shape)

    #  as you can see result is = concat of all different parts. because it concat, it looks like one hot encoded, but not really. it has multiple 1's in one row.
    # so i chatGPT how to process this data for embedding. like i did for chatGPT GPT models. (from andrej youtube)
    result=torch.cat([dna,target,pbs,rtt,del_target,ins_dna,sub_dna],dim=1)
    # print("\n>> result.shape= ", result.shape)


    total_x.append(result)

    y_true=df['Measured_PE_efficiency'][i]
    y_true
    total_y.append(y_true)
    print("i: ",i)
#%%
print(total_y)
#%%

x=torch.stack(total_x,dim=0)
#%%
x.shape
#%%
total_y=[torch.tensor(item) for item in total_y]
#%%
y=torch.stack(total_y,dim=0)
print("\n>> y.shape= ", y.shape)
#%%

# save below
# torch.save({
#     'x': x,
#     'y': y,
# }, 'data_checkpoint.pth')

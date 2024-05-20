#%%

'''
# del 2 (15000)
TTCTTGTCTTCCTCAGGAGCCGTGTGGACAGGGGACAACACTGCCGAGTGGGACCATTTGAAGATCTCTATTCC
TTCTTGTCTTCCTCAGGAGCC|GTGTGGACAGGGGACAACACTGCCGAGTGGGACCATTTGAAGATCTCTATTCC
xxxxxxxxTTCCTCAGGAGCC|GTGTGGAC{{GGGACAACACTGCCGAGTGGGxxxxxxxxxxxxxxxxxxxxxxxx

# del 2
AGTTTTTATGTGGATCAATATCCTCGGAAAGAATTAGACTGTTATTTATGTAGGCCTTGATAAGAGTCTGATTT
xxxxxxxxxxxxxxxCAATATCCTCGGAAAGAATTAGACTTATxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

AGTTTTTATGTGGATCAATAT|CCTCGGAAAGAATTAGACTGTTATTTATGTAGGCCTTGATAAGAGTCTGATTT
xxxxxxxxxxxxxxxCAATAT|CCTCGGAAAGAATTAGACT{{TATxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# insert
GGCGGATGAAGCGGGAGATGATGGGGGGCCGCAGCAGGTTCTGAACCGTGGAGGTGCGGTCCGAGTTGCCATAG
GGCGGATGAAGCGGGAGATGA|TGGGGGG{CCGCAGCAGGTTCTGAACCGTGGAGGTGCGGTCCGAGTTGCCATAG
xxxxxxxxxxGCGGGAGATGA|TGGGGGGGCCGCAGCAGGTTCTGAACCGTxxxxxxxxxxxxxxxxxxxxxxxx

# insert 2 
TAGCCTTTTGTGTGAAAGAAGAGAAGGGGAGCACGATGCTAGTAGACGCCTAAAAACTGAATTTCTAATAGAAT
TAGCCTTTTGTGTGAAAGAAG|AGAAGGGG}}AGCACGATGCTAGTAGACGCCTAAAAACTGAATTTCTAATAGAAT
xxxxxxxxxxxxTGAAAGAAG|AGAAGGGGGGAGCACGATGCTAGTAGxxxxxxxxxxxxxxxxxxxxxxxxxxx

#sub
TCAAGAGCCCAGAGCTTCAGGCCGAGGCCAAGTAAGTCTCAGGGCAAGGGGTTCAGGGGCTGTGGAACTGTGGA
xxxxxxxxxxAGAGCTTCAGGCCGAGGCCAAATxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

TCAAGAGCCCAGAGCTTCAGG|CCGAGGCCAA{G}TAAGTCTCAGGGCAAGGGGTTCAGGGGCTGTGGAACTGTGGA
xxxxxxxxxxAGAGCTTCAGG|CCGAGGCCAA{A}Txxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# sub 2 (285k)
CTCTGAGACTTCTGCCCAGACTTCAGGAATGAAGTGTTATTTTCCACTGGTGATCTGATCCTTCAGTGACACTT
xxxxxxxxxxxxTGCCCAGACTTCAGGAATGAAGTTGTATTTTCCACTGGTGATCTGATCCTTCxxxxxxxxxx
CTCTGAGACTTCTGCCCAGAC|TTCAGGAATGAAGT{GT}TATTTTCCACTGGTGATCTGATCCTTCAGTGACACTT
xxxxxxxxxxxxTGCCCAGAC|TTCAGGAATGAAGT{TG}TATTTTCCACTGGTGATCTGATCCTTCxxxxxxxxxx

'''

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
max_length
# %%


#%%
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

total_list=[]
TEMP=1000
for i in tqdm(range(len(df))[:TEMP]):
    # print("df['type_del'][i]: ",df['type_del'][i])
    # break


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

    edit_pos




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


    type1

    DNA


    TARGET

    print("\n>> len(DNA)= ", len(DNA))
    print("\n>> len(TARGET)= ", len(TARGET))



# print("\n>> encoded_array.shape= ", encoded_array.shape)


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



    list1=[]
    for j in range(len(TARGET)):
        dna=DNA[j]
        tar=TARGET[j]
        if dna=='g' and tar=='g':
            list1.append(0)
        elif dna=='g' and tar=='a':
            list1.append(1)
        elif dna=='g' and tar=='c':
            list1.append(2)
        elif dna=='g' and tar=='t':
            list1.append(3)
        # elif dna=='g' and tar=='g':
        
        elif dna=='a' and tar=='a':
            list1.append(4)
        elif dna=='a' and tar=='c':
            list1.append(5)
        elif dna=='a' and tar=='t':
            list1.append(6)
        elif dna=='a' and tar=='g':
            list1.append(7)
        
        elif dna=='t' and tar=='a':
            list1.append(8)
        elif dna=='t' and tar=='c':
            list1.append(9)
        elif dna=='t' and tar=='t':
            list1.append(10)
        elif dna=='t' and tar=='g':
            list1.append(11)
        
        elif dna=='c' and tar=='a':
            list1.append(12)
        elif dna=='c' and tar=='c':
            list1.append(13)
        elif dna=='c' and tar=='t':
            list1.append(14)
        elif dna=='c' and tar=='g':
            list1.append(15)
        
        elif dna=='a' and tar=='x':
            list1.append(16)
        elif dna=='c' and tar=='x':
            list1.append(17)
        elif dna=='t' and tar=='x':
            list1.append(18)
        elif dna=='g' and tar=='x':
            list1.append(19)
        
        elif dna=='a' and tar=='d':
            list1.append(20)
        elif dna=='c' and tar=='d':
            list1.append(21)
        elif dna=='t' and tar=='d':
            list1.append(22)
        elif dna=='g' and tar=='d':
            list1.append(23)

        elif dna=='d' and tar=='a':
            list1.append(24)
        elif dna=='d' and tar=='c':
            list1.append(25)
        elif dna=='d' and tar=='t':
            list1.append(26)
        elif dna=='d' and tar=='g':
            list1.append(27)
        elif dna=='x' and tar=='x':
            list1.append(28)
            # print(dna,tar)

    print("\n>> len(list1)= ", len(list1))

    list1
    num_features = 29

    def one_hot_encode(indices, num_features):
        # Create an array of zeros with shape (number of indices, number of features)
        one_hot_encoded = torch.zeros((len(indices), num_features), dtype=torch.int64)
        
        # Set the appropriate element to 1
        for i, idx in enumerate(indices):
            one_hot_encoded[i, idx] = 1
        
        return one_hot_encoded

    # Encoding the list
    encoded_array = one_hot_encode(list1, num_features)
    print(encoded_array)
    total_list.append(encoded_array)




    y_true=df['Measured_PE_efficiency'][i]
    y_true
    total_y.append(y_true)


    # break
#%%
print("\n>> encoded_array.shape= ", encoded_array.shape)

#%%

x=torch.stack(total_list,dim=0)
print("\n>> x.shape= ", x.shape)

total_y=[torch.tensor(item) for item in total_y]
#%%
y=torch.stack(total_y,dim=0)
print("\n>> y.shape= ", y.shape)
#%%

### save below

# torch.save({
#     'x': x,
#     'y': y,
# }, 'data/data_with_feature_29.pth')

# %%

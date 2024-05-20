#%%
from datetime import datetime,timedelta
import sys,os,copy,ast,socket,random,math,webbrowser,getpass,time,shutil,ast,subprocess,requests
import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt
def print2(*args):
    formatted_args = []
    for arg in args:
        if isinstance(arg, float):  # Check if the argument is a float
            formatted_arg = f'{round(arg, 3):.3f}'  # Round and format the float
        else:
            formatted_arg = str(arg)
        formatted_args.append(formatted_arg)
    print('\t'.join(formatted_args))

def reset_index(df):
    df=df.reset_index()    
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])
    return df

def read_excel(path):
    if path.endswith(".csv"):

        df=pd.read_csv(path,index_col=0)
    else:
        df=pd.read_excel(path,index_col=0)
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])    
    return df


def dump1(*varnames):
    python_script=os.path.basename(__file__)
    if not os.path.exists(f't7_print/{python_script}'):
        os.makedirs(f't7_print/{python_script}')
    global_vars = globals()
    string1=''
    for varname in varnames:
        if varname in global_vars:
            dict1={varname: global_vars[varname]}
            string1=string1+str(dict1)+', '
    print(string1)
    filename=string1
    with open(f't7_print/{python_script}/{filename}','w') as f:
        f.write(string1)

def print1(content_dict,filename):
    if '.txt' not in filename:
        filename=filename+'.txt'
    path='data/log/'
    if not os.path.exists(path):
        os.makedirs(path)
    fullpath=path+filename
    timedelta1=0
    timedelta2=100
    script_name = os.path.basename(__file__)
    today1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1)).strftime("%Y-%m-%d %H:%M:%S"))
    today2=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1))).timestamp()
    past1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta2)).strftime("%Y-%m-%d"))
    past2=datetime.strptime('20210218', "%Y%m%d").timestamp()

    with open(fullpath,'a+') as f:
        f.write(str(content_dict))
        f.write(f' <-- {today1} {script_name} -->')   
        f.write('\n') 



def list_minus(list1, list2):
    # Convert lists to sets and find the difference
    set_difference = set(list1) - set(list2)
    # Convert the set back to a list
    return list(set_difference)



def base_list_save(filename,list1):
    
    content=str(list1)

    if filename.endswith('txt'):
        filename=filename.replace('.txt','')
    path='log/lists'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'log/lists/{filename}.txt','w') as f:
        f.write(content)

def base_list_collections():
    return os.listdir('log/lists')

def base_list_read(filename):
    with open(f'log/lists/{filename}.txt','r') as f:
        data=f.readline()
    data=ast.literal_eval(data)
    return data



def csv_update_insert_one(database_name,collection_name,dn,no_duplicate_column):
    

    # csv_update_insert_one
    if 'cjs' in getpass.getuser():
        path3='C:/Users/cjsdl/Documents/data/'
    elif 'linux' in getpass.getuser():
        path3=f'/home/{getpass.getuser()}/Documents/data/'
    else:
        path3=f'/Users/{getpass.getuser()}/Documents/data/'
    
    path=path3+database_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)



    fullpath=path+collection_name+'.csv'
    try:
        do=read_excel(fullpath)
        
        do.set_index(no_duplicate_column, inplace=True)
        dn.set_index(no_duplicate_column, inplace=True)
        # print("dn: ",dn)
        # Update 'do' with the values from 'dn' where the indices match, then concatenate the rest of 'dn'
        do.update(dn)
        do = pd.concat([do, dn[~dn.index.isin(do.index)]])

        # Reset the index if needed
        do.reset_index(inplace=True)
        # print("do: ",do)        
        do

    except:
        # do=pd.DataFrame()
        do=dn

    do.to_csv(fullpath)
    # print('success csv')
    return do


def open_excel(df):
    random1=random.randint(1,10)

    temp_file=f'temp_sample_{random1}.xlsx'
    if 'cjsdl' in getpass.getuser():
        export_path='G:/My Drive/ibkr/1_total_data/8 temp/'
        print('start - to_excel a large file maybe ...')
        df.to_excel(export_path+temp_file)
        print('success - to_excel a large file maybe ...')
        excel1=fr"G:\My Drive\ibkr\1_total_data\8 temp\temp_sample_{random1}.xlsx"
        # r"C:\Users\cjsdl\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)\Anaconda Prompt (miniconda3).lnk"
        os.startfile(excel1)
        
    else:
        export_path=f'/Users/{getpass.getuser()}/Library/CloudStorage/GoogleDrive-ryan.ichun9@gmail.com/My Drive/ibkr/1_total_data/8 temp/'
        print('start - to_excel a large file maybe ...')

        df.to_excel(export_path+temp_file)
        os.system("open -a 'Microsoft Excel.app' '{}{}'".format(export_path,temp_file))
# %%


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # value for the last update
        self.avg = 0  # average of accumulated values
        self.sum = 0  # sum of accumulated values
        self.count = 0  # number of accumulated values

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Early_stop():
    def __init__(self,step):
        self.losses=[]
        self.step=step
        self.max=float("inf")
    def update(self,loss):
        if len(self.losses)>self.step:
            self.losses.pop(0)
            self.losses.append(loss)
            min1=min(self.losses)
            if min1<=self.max:
                min1=min(self.losses)
                self.max=min1
                return False
            else:
                return True
        
        self.losses.append(loss)
        return False

def work():
    with open('log/test.txt','r') as f:
        data=f.readline()
        if 'work'   in data:
            return True
        else:
            return False
def show():
    with open('log/test.txt','r') as f:
        data=f.readline()
        if 'show'   in data:
            return True
        else:
            return False
def save():

    with open('log/test.txt','r') as f:
        data=f.readline()
        if 'save'   in data:
            return True
        else:
            return False


def print_tab(*args):
    # Initialize an empty list to hold formatted strings
    formatted_items = []
    
    # Iterate through each argument provided
    for item in args:
        # Check if the item is a float
        if isinstance(item, float):
            # Format the float to 4 significant figures
            formatted_item = f"{item:.4g}"
        else:
            # Convert the item to string if it's not a float
            formatted_item = str(item)
        
        # Append the formatted string to the list
        formatted_items.append(formatted_item)
    
    # Join all formatted items with a tab separator and print the result
    print("\t".join(formatted_items))



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
import random
import numpy as np
import pandas as pd
if torch.backends.mps.is_available():
    device='mps'
elif torch.cuda.is_available():
    device='cuda'
else:
    device="cpu"

DEVICE=device

def random_seed(m):
    random.seed(m)
    torch.manual_seed(m)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(m)
        torch.cuda.manual_seed_all(m)
    np.random.seed(m)
random_seed(1)
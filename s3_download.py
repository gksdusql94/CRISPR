
import subprocess,os
def aws_download_everything(target_folder,to_folder,EXCLUDE=False,DELETE=False,exclude_list=[]):
    exclude_list = [element for element in exclude_list if element != '']
    target_folder = target_folder.rstrip('/') + '/'
    to_folder = to_folder.rstrip('/') + '/'
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)
    command =f"aws s3 sync s3://bluerand3/{target_folder} {to_folder} --exclude 'ignore/*'"
    for item in exclude_list:
        command=command+f" --exclude '{item}/*'"
    if EXCLUDE:
        command=command+" --exclude 'data/*'"
    if DELETE:
        command=command+' --delete'
    subprocess.run(command, shell=True)
    print('success')
to_folder=os.getcwd()
target_folder='bio/crispr_ml_v3'

list1=[
    'storage',
    '',
    '',
    '',
    '',
    '',

]
aws_download_everything(target_folder,to_folder,EXCLUDE=True,exclude_list=list1)

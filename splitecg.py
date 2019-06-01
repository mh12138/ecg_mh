import numpy as np
import wfdb
import os

import shutil
not_list2 = ('100','102','103','104','114','117','123','124')
not_list1 = ('102','104','114')
AAMI_label = [{'N','.','L','R','e','j'},{'A','a','J','S'},{'V','E'},{'F'},{'f','/','Q'}]
AAMI_label_name = ['N','S','V','F','Q']
file_list = []
data_dir = 'mit_database/'

handle_root = 'ECGI_data1/'
index_root = 'ECGI_index_dataset1/'
if(os.path.exists(index_root)):
    pass
else:
    os.mkdir(index_root)

if(os.path.exists(handle_root)):
    pass
else:
    os.mkdir(handle_root)

list_train_path = os.path.join(index_root,'trainlist.txt')
list_test_path = os.path.join(index_root,'testlist.txt')
fp1 = open(list_train_path ,'w')
fp2 = open(list_test_path ,'w')
fp0 = open(os.path.join(index_root,'all.txt'),'w')

def get_aami_label_number(s1):
    for idconf , i in enumerate(AAMI_label):
        if(s1 in i):
            return idconf
    return -1

def convert_singal(oral_single , downsampled = 64):
    len_t = len(oral_single)
    oral_r = np.linspace(1,len_t,len_t)
    now_r = np.linspace(1,len_t , downsampled)
    return np.interp(now_r , oral_r  , oral_single)

def  writerecord(save_root,savefilename , writearray):

    ftp = open(os.path.join(save_root,savefilename),'w')
    for ecg in writearray:
        for line in ecg:
            ftp.write('{:.4f} '.format(line))
        ftp.write('\n')
    ftp.close()

for fname in os.listdir(data_dir):
    words = fname.split('.')
    if(words[0] in file_list):
        continue
    else:
        print(words[0])
        file_list.append(words[0])
        tfilename = os.path.join(data_dir , words[0])
        s1 = wfdb.rdsamp(tfilename)
        ann = wfdb.rdann(tfilename , 'atr')
        if(words[0] in not_list1):
            continue
        # else:
        #     print(s1[1]['sig_name'])
        II = np.array(s1[0][:,0])
        V1 = np.array(s1[0][:,1])
        label_pos = ann.sample
        label_name = ann.symbol
        # print(s1[1]['sig_name'])
        # if(s1[1]['sig_name'][0] != 'MLII'):
        #     print('##')
        # break
        # print(os.path.join(handle_root,'{}_train.txt'.format(words[0])))

        for idk , ikey in enumerate(label_pos):

            if(idk < len(label_pos) -5):
                tlabel = get_aami_label_number(label_name[idk + 1])
                if (tlabel == -1):
                    continue
                onset = (ikey + label_pos[idk+1]) //2
                offset = (label_pos[idk+1]+ label_pos[idk+2])//2
                newII = convert_singal(II[onset: offset])
                newV1 = convert_singal(V1[onset: offset])
                onefilename = '{}_{}.txt'.format(words[0],idk)

                randomx = np.random.randint(0,10)
                cpath = os.path.join(handle_root , '{}_{}.txt'.format(words[0],idk))


                fp0.write('{} {}\n'.format(cpath,tlabel))
                if randomx <= 7:
                    fp1.write('{} {}\n'.format(cpath,tlabel))
                else:
                    fp2.write('{} {}\n'.format(cpath, tlabel))
                fp0.write('{} {}\n'.format(cpath, tlabel))
                writerecord(handle_root,onefilename,[newII,])


fp0.close()
fp1.close()
fp2.close()


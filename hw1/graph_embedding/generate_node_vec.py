import pandas as pd
import numpy as np
import torch
import pickle
import os

DIM_NODE2VEC = 128
PATH_NODE2VEC = 'cit_128.emb'
#ID_SRC = '../data/task2_trainset.csv'  # generating node_vec for train/valid set
ID_SRC = '../data/task2_public_testset.csv'  # generating node_vec for test set



# read the vectors and corresponding paperid's
if os.path.exists('paperid_vec.npy') and os.path.exists('paperid.npy'):
    paperid_vec = np.load('paperid_vec.npy')
    paperid = np.load('paperid.npy')
    print('npy files loaded.')
else:
    cit_emb = np.loadtxt(PATH_NODE2VEC, delimiter=' ', skiprows=1)
    paperid_vec = cit_emb[:, 1:]
    paperid = cit_emb[:, 0].astype(np.uint32)
    np.save('paperid_vec', paperid_vec)
    np.save('paperid', paperid)
    print('npy files saved.')

DIM_NODE2VEC = paperid_vec.shape[1]

print(paperid_vec.shape)
print(paperid.shape)
print(paperid_vec[:5, :5])
print(paperid[:10])



# generate or load the paperid-tensor dict.
if os.path.exists('paperid2vec_dict.pkl'):
    with open('paperid2vec_dict.pkl', 'rb') as file:
        paperid2vec_dict = pickle.load(file)
        print('pkl loaded.')
else:
    # build a paperid-tensor dict
    paperid2vec_dict = {}
    _total_samples_num = paperid_vec.shape[0]
    for index, id in enumerate(paperid):
        print(f'converting... {index}/{_total_samples_num}', end='\r')
        paperid2vec_dict[id] = torch.from_numpy(paperid_vec[index]).reshape((1, DIM_NODE2VEC))
    print('\n')

    with open('paperid2vec_dict.pkl', 'wb') as file:
        pickle.dump(paperid2vec_dict, file)

print(len(paperid2vec_dict))
print(paperid2vec_dict[1671395369])



# mapping id with tensor
dataset_id = pd.read_csv(ID_SRC, dtype=str)['Id']  # list of id's
id_paperId = dict(pd.read_csv('id_paperId.tsv', sep='\t').values)  # id to paperId
print('id_paperId.shape: ', len(id_paperId))

node_vec = torch.zeros((0, DIM_NODE2VEC)).type(torch.DoubleTensor)
_total_samples_num = dataset_id.shape[0]
hit, miss = 0, 0
for index, id in enumerate(dataset_id):
    print(f'dealing with {index}/{_total_samples_num}...', end='\r')
    if id in id_paperId:
        PaperId = id_paperId[id]
        if PaperId in paperid2vec_dict:
            node_vec = torch.cat((node_vec, paperid2vec_dict[PaperId]), 0)
            hit += 1
        else:
            node_vec = torch.cat((node_vec, torch.zeros((1, DIM_NODE2VEC)).type(torch.DoubleTensor)), 0)
            miss += 1
    else:
        node_vec = torch.cat((node_vec, torch.zeros((1, DIM_NODE2VEC)).type(torch.DoubleTensor)), 0)
        miss += 1
print('\n')
print('final tensor shape: ', node_vec.shape)
print(node_vec[:30, :5])
print('hit: ', hit)
print('miss: ', miss)
print('total:', hit+miss)


# saving the node vector
with open('node_vec.pkl', 'wb') as file:
    pickle.dump(node_vec, file)
print('node_vec.pkl saved.')
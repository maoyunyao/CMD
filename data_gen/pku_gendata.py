import argparse
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

import numpy as np
import os

def read_data(data_path, name, max_body=4, num_joint=25):  # top 2 body
    filename, action_idx = name.split('_')
    action_idx = int(action_idx)
    seq_data = np.loadtxt('{}/skeleton/{}'.format(data_path, filename))
    label = np.loadtxt('{}/label/{}'.format(data_path, filename), delimiter=',')
    start, end = int(label[action_idx][1]), int(label[action_idx][2])
    
    data = seq_data[start: end, :]  # num_frames * 150
    data = data.reshape(data.shape[0], 2, 25, 3)  # num_frame, num_body, num_joint, xyz
    data = data.transpose(3, 0, 2, 1)  # xyz, num_frame, num_joint, num_body
    return data

def gendata(data_path, out_path, benchmark='xview', part='eval'):
    # Read cross_subject_v2.txt and cross_view_v2.txt to obtain training_views training_subjects
    with open('{}/cross_view_v2.txt'.format(data_path), 'r') as f:
        lines = f.readlines()
        training_views = lines[1].strip('\n').split(', ')
    with open('{}/cross_subject_v2.txt'.format(data_path), 'r') as f:
        lines = f.readlines()
        training_subjects = lines[1].strip('\n').split(', ')


    sample_name = []
    sample_label = []
    for filename in os.listdir('{}/skeleton'.format(data_path)):
        if benchmark == 'xview':
            istraining = (filename[:-4] in training_views)
        elif benchmark == 'xsub':
            istraining = (filename[:-4] in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            label = np.loadtxt('{}/label/{}'.format(data_path, filename), delimiter=',')
            for idx in range(label.shape[0]):
                sample_name.append('{}_{}'.format(filename, str(idx)))
                sample_label.append(label[idx][0] - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fl = open_memmap(
        '{}/{}_num_frame.npy'.format(out_path, part),
        dtype='int',
        mode='w+',
        shape=(len(sample_label),))

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_data(data_path, s, max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:min(data.shape[1], max_frame), :, :] = data[:, 0:min(data.shape[1], max_frame), :, :]  # num_frame 太大会截断！
        fl[i] = data.shape[1] # num_frame

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PKU-MMD-v2 Data Converter.')

    parser.add_argument('--data_path', default='/data/user/dataset/PKU-MMD/v2/')
    parser.add_argument('--out_folder', default='../data/PKU-MMD-v2-AGCN/')
    benchmark = ['xsub','xview', ]

    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                benchmark=b,
                part=p)

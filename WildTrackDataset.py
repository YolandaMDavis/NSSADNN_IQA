import os

import torch

import IQADataset
import pandas
import cv2
import numpy as np
from torch.utils.data import Dataset

from brisque import brisque
from niqe import niqe
from piqe import piqe


class WildTrackDataset(Dataset):

    def __init__(self, dataset_file, config, status):
        self.gray_loader = IQADataset.gray_loader
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        images = pandas.read_csv(dataset_file,header=None,names=["image_dir","species","class","image_filename","rating"])
        self.row_count = images.shape[0]
        categories = {'Bongo':0, 'Cheetah':1, 'Elephant':2, 'Jaguar':3, 'Leopard':4, 'Lion':5, 'Otter':6, 'Panda':7, 'Puma':8, 'Rhino':9, 'Tapir':10, 'Tiger':11}

        # get rating
        self.mos = images["rating"].to_numpy()
        self.mos = np.select([(self.mos < 4), (self.mos >= 4)],[0,1])
        self.features = []
        self.label = []

        for index, row in images.iterrows():
            print("Processing file number:" + str(index))
            file_path = os.path.join(row['image_dir'], row['image_filename'])
            im = self.gray_loader(file_path)
            im_features = cv2.imread(file_path)
            im_features = cv2.cvtColor(im_features, cv2.COLOR_BGR2RGB)
            brisque_features = brisque(im_features)
            n_score = niqe(im_features)
            p_score,_,_,_ = piqe(im_features)
            other_scores = [n_score, p_score, categories[row['species']]]
            full_features = np.append(brisque_features,np.array(other_scores))
            self.label.append(self.mos[index])
            self.features.append(full_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.Tensor([self.label[idx]]), torch.Tensor(self.features[idx]))
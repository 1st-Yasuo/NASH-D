import os
import pandas as pd
from PIL import Image
from glob import glob
import random
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, utils, models
from torch.utils.data import DataLoader, Dataset, random_split



class SteatosisDatasets(Dataset):
    def __init__(self, imgs_dir, labels, seed=0, train=True, testdata_percents=0.2, img_size=(224,224),mean,std):
        self.train = train
        self.imgs_dir = imgs_dir
        self.wsi_ids = os.listdir(imgs_dir)
        self.wsi_dirs = glob(imgs_dir+'*')
        self.labels_df = pd.read_excel(labels).set_index('HE_ID')
        self.test = testdata_percents
        self.seed = seed
        self.train_ids, self.test_ids = self.split_dataset()
        self.size = img_size
        self.mean = mean
        self.std = std

    def split_dataset(self):
        train_ids = self.wsi_ids
        test_ids = []
        for x in range(4):
            random.seed(self.seed)
            test_idx = random.sample(list(self.labels_df[self.labels_df['S_score'] == x].index), int(len(
                self.labels_df[self.labels_df['S_score'] == x])*self.test))
            test_ids.extend(test_idx)

        for i in test_ids:
            train_ids.remove(str(i))
        train_ids = list(map(str, train_ids))
        test_ids = list(map(str, test_ids))
        return train_ids, test_ids

    def __len__(self):
        n = 0
        if self.train:
            for i in self.train_ids:
                n += len(os.listdir(self.imgs_dir+i+'/'))
        if not self.train:
            for i in self.test_ids:
                n += len(os.listdir(self.imgs_dir+i+'/'))
        return int(n)

    def __getitem__(self, i):

        transform = transforms.Compose([transforms.Resize(
            self.size), transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        if self.train:
            train_dirs = []
            for idx in self.train_ids:
                train_dirs.extend(
                    glob(os.path.join(self.imgs_dir, idx)+'/*.jpg'))
            #index = train_dirs[i].split('/')[-1].split('.')[0]
            img = Image.open(train_dirs[i])
            img_tensor = transform(img)
            wsi_idx = train_dirs[i].split('/')[-2]
            # print(len(train_dirs))
            label = np.array(self.labels_df.loc[int(wsi_idx)][0])
        if not self.train:
            test_dirs = []
            for idx in self.test_ids:
                test_dirs.extend(
                    glob(os.path.join(self.imgs_dir, idx)+'/*.jpg'))
            img = np.array(Image.open(test_dirs[i]))
            wsi_idx = test_dirs[i].split('/')[-2]
            label = np.array(self.labels_df.loc[int(wsi_idx)][0])
        return (img_tensor, torch.from_numpy(label).type(torch.FloatTensor))

    # @classmethod

    def regularization(self, w=224, h=224):
        train_ids, test_ids = self.split_dataset()
        means, stdevs = [], []
        arr_list = []
        slide_dirs = [self.imgs_dir + train_ids[i] +
                      '/*.jpg' for i in range(len(train_ids))]
        patches_dirs = []
        for d in slide_dirs:
            patches_dirs.extend(glob(d))
        print(len(patches_dirs))
        i = 0
        for image_name in patches_dirs:
            img = Image.open(image_name)
            img = img.resize((w, h))
            arr = np.array(img)
            arr = arr[:, :, :, np.newaxis]
            arr_list.append(arr)
            i += 1
            print(i, '/', len(patches_dirs))
        matrix = np.concatenate(arr_list, axis=3).astype(np.float32)/255.
        for i in range(3):
            channel_values = matrix[:, :, i, :].ravel()  # ravel 节省内存
            means.append(np.mean(channel_values))
            stdevs.append(np.std(channel_values))
        return means, stdevs


if __name__ == '__main__':
    # mean,std = dataset.regularization()
    dirname = ''
    label_excel = ''
    image_datasets = SteatosisDatasets(dirname,
                                       label_excel,
                                       seed=0, train=True)
    n_val = int(len(image_datasets) * 0.25)
    n_train = len(image_datasets) - n_val
    train, val = random_split(image_datasets, [n_train, n_val])
    train_dataloader = DataLoader(
        dataset=train, batch_size=16, num_workers=8, shuffle=True,pin_memory=True)
    val_dataloader = DataLoader(
        dataset=val, batch_size=16, num_workers=8, shuffle=True,pin_memory=True)



import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import os
from collections import defaultdict
import pandas as pd
import torch.nn.functional as F
from glob import iglob


def prediction(img_name, model):
    """[summary]

    Args:
        img_name ([type]): [description]
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
    img = Image.open(img_name)
    trans = transforms.Compose([transforms.Resize(
        [224, 224]), transforms.ToTensor(),
        #transforms.Normalize([0.7776364, 0.44906333, 0.79358804], [
        #    0.12817128, 0.22058904, 0.100848824])

    ])
    tensor = trans(img)
    if torch.cuda.is_available():
        tensor.cuda()
    tensor = Variable(torch.unsqueeze(
        tensor, dim=0).float(), requires_grad=False).to('cuda')
    model.eval()
    y_pred = model(tensor)
    label = torch.argmax(y_pred).to('cpu')
    label = label.numpy().tolist()
    return y_pred, label


def vote(label_list, mode='mean'):
    """[summary]

    Args:
        label_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    if mode == 'mode':
        counts = np.bincount(label_list)
        score = np.argmax(counts)
    elif mode == 'mean':
        mean_value = np.mean(label_list)
        if mean_value < 0.2:
            score = 0
        elif 0.15 < mean_value <= 1.1:
            score = 1
        elif 1.1 < mean_value <= 2:
            score = 2
        elif 2 < mean_value <= 3:
            score = 3

    return score


def run(dirname):
    n = 0
    predict_data = {}
    for i in os.listdir(dirname):
        sample_list = os.path.join(dirname, i)
        label_list = np.array([], dtype=int)
        for j in os.listdir(sample_list):
            img_name = os.path.join(sample_list, j)
            img = Image.open(img_name)
            img_arr = np.array(img)
            white_area = np.sum(np.all(img_arr > [250, 250, 250], axis=2))
            resolution = img_arr.shape[0]*img_arr.shape[1]
            if white_area/resolution > 0.5:
                continue
            label = prediction(img_name, model)
            print('{} is predicted as {}'.format(j.split('.')[0], label))
            label_list = np.append(label_list, label)
        sample_level = vote(label_list, mode='mode')
        n += 1
        print('the {} sample: {} is predicted as {}'.format(n, i, sample_level))
        print('-----------------------')
        predict_data[f'{i}'] = sample_level
    df = pd.Series(predict_data)
    return df


def softmax_vote():
    maindir = '/home/lisj/Documents/Datasets/5x_split_images/'
    model_path = '/home/lisj/Desktop/steatosis_classifier/res50.pth'
    model = torch.load(model_path)
    model.to('cuda')
    data = []
    index = []
    for i,idx in enumerate(iglob(maindir+'*')):
        res_list = torch.Tensor()
        n = idx.split('/')[-1]
        index.append(n)
        for patch in iglob(f'{idx}/*.jpg'):
            predict, t = prediction(patch, model)
            res = F.softmax(predict, dim=1)
            res = res.to('cpu')
            res_list = torch.cat([res_list, res], dim=0)
        values_list = torch.sum(res_list, dim=0).detach().numpy().tolist()
        data.append(values_list)
        print(f'the {i} : {n} is done!')
    print(len(data))
    df = pd.DataFrame(data,index=index)
    df.to_csv('../res50_softmax_regression.csv')


def counts_regression():
    maindir = '/home/lisj/Documents/Datasets/5x_split_images/'
    model_path = '/home/lisj/Desktop/steatosis_classifier/res50.pth'
    model = torch.load(model_path)
    index = [None]
    df = pd.Series(index=[0, 1, 2, 3], dtype=int)
    for i,idx in enumerate(iglob(maindir+'*')):
        res_list = []
        n = idx.split('/')[-1]
        index.append(n)
        for patch in iglob(f'{idx}/*.jpg'):
            predict, label = prediction(patch, model)
            res_list.append(label)
        counts_dic = pd.value_counts(res_list).sort_index()
        df = pd.concat([df, counts_dic], axis=1)
        print(counts_dic)
        print(f'the {i} : {n} is done!')
    df.columns = index
    df = df.T
    df.to_csv('../res50_counts_data.csv')  


if __name__ == '__main__':

    #dirname = '/home/lisj/Documents/Datasets/5x_split_images/'
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = torch.load('/home/lisj/Desktop/steatosis_classifier/2021_4_29_dense201.pth',
    #                   map_location=torch.device(device))
    # model.eval()
    #df = run(dirname)
    # df.to_excel('/home/lisj/Desktop/workspace/wsi_prediction_dense201_mode.xlsx')
    softmax_vote()
    #counts_regression()

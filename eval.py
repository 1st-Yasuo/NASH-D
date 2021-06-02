import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
from datasets import get_dataset


def test_eval(model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dir = r'/home/lisj/Documents/Datasets/liver_steatosis_datasets/test/'
    test_data, _ = get_dataset(test_dir, 0, batch_size=16, shuffle=False)
    labels_list = torch.tensor([]).to(device=device)
    prediction = torch.tensor([]).to(device=device)
    for imgs, labels in test_data:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        print(labels)
        label_pred = model(imgs)
        _, pred = torch.max(label_pred.data, 1)
        labels_list = torch.cat((labels_list, labels))
        prediction = torch.cat((prediction, pred), 0)
    df = pd.DataFrame()
    labels_list = labels_list.to(device='cpu')
    prediction = prediction.to(device='cpu')

    df['test_labels'] = labels_list.numpy()
    df['prediction'] = prediction.numpy()

    return df


def prediction(img, model):
    img = Image.open(img)
    label_pred = model(img)

if __name__ == '__main__':
    model = torch.load('/home/lisj/Desktop/steatosis_classifier/res50.pth')
    data = test_eval(model)
    #data.to_excel('../weakly_dense201_directly_test.xlsx')


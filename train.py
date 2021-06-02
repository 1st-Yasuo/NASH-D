import os
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import sys
import time
import argparse
from torch.utils.data import DataLoader, datasets, random_split
from datasets import SteatosisDatasets


os.environ['NUMEXPR_MAX_THREADS'] = '16'

def get_args():
    parser = argparse.ArgumentParser(
        description='Train the Model on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d',
                        '--dirname',
                        type=str,
                        help='Traindata directory path',
                        dest='dir')
    parser.add_argument('-e',
                        '--epochs',
                        metavar='E',
                        type=int,
                        default=10,
                        help='Number of epochs',
                        dest='epochs')
    parser.add_argument('-b',
                        '--batch-size',
                        metavar='B',
                        type=int,
                        nargs='?',
                        default=16,
                        help='Batch size',
                        dest='batchsize')
    parser.add_argument('-l',
                        '--learning-rate',
                        metavar='LR',
                        type=float,
                        nargs='?',
                        default=0.001,
                        help='Learning rate',
                        dest='lr')
    parser.add_argument(
        '-v',
        '--validation',
        dest='val',
        type=float,
        default=25.0,
        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


def get_dataset(data_dir, val_percent=0.25, batch_size=16, num_workers=8, shuffle=True):
    image_datasets = SteatosisDatasets('/home/lisj/Documents/split_images/5x_splitimages/',
                                '/home/lisj/Documents/split_images/corresbonding_labels.xlsx',
                                seed=0,train=True)
    n_val = int(len(image_datasets) * val_percent)
    n_train = len(image_datasets) - n_val
    train, val = random_split(image_datasets, [n_train, n_val])
    # print(image_datasets.class_to_idx)
    train_dataloader = DataLoader(
        dataset=train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    val_dataloader = DataLoader(
        dataset=val, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    return train_dataloader, val_dataloader


def model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048, 1000), nn.Linear(1000, 4))
    
    return model


def _eval(model, dataset, batch_size):
    model.train(False)
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    acc = 0
    for imgs, labels in dataset:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        # logging.info(f'''{imgs.shape}''')
        labels_pred = model(imgs)
        val_loss += criterion(labels_pred, labels.long())
        _, pred = torch.max(labels_pred.data, 1)
        acc += torch.sum(pred == labels.long())
    return val_loss / len(dataset), acc / (len(dataset) * batch_size)


def train_model(dirname,
                model,
                device,
                epochs=10,
                lr=0.001,
                batch_size=16,
                val_percent=0.25):
    # load data
    train_data, valid_data = get_dataset(dirname, batch_size=batch_size)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

    logging.info(f'''
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_data)*batch_size}
        Validation size: {len(valid_data)*batch_size}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(epochs):
        logging.info(f'---the {epoch+1} epoch---')
        model.train(True)
        train_loss = 0.0
        train_corrects = 0
        for batch, data in enumerate(train_data, 1):
            imgs, labels = data
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            labels_pred = model(imgs)
            _, pred = torch.max(labels_pred, 1)
            optimizer.zero_grad()
            #print(labels)
            loss = criterion(labels_pred, labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_corrects += torch.sum(pred == labels)
            # logging.info(
            #    f'loss:{loss.item():.4f}, ACC:{torch.sum(pred == labels) / batch_size:.4f}'
            # )

            if batch % 50 == 0:
                logging.info(
                    f'Batch {batch}, Train loss: {train_loss/batch: .4f}, Train ACC: {100*train_corrects/(batch_size*batch): .4f} %'
                )
                # writer.add_scalar('learning_rate',
                #                  optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
        writer.add_scalars('Loss&ACC/Train', {
            'Loss': train_loss/batch,
            'ACC': train_corrects/(batch_size*batch)
        }, global_step)
        with torch.no_grad():
            val_loss, val_ACC = _eval(model, valid_data, batch_size)
            logging.info(
                f'Val loss: {val_loss:.4f}, Val ACC:{val_ACC:.4f}'
            )
        # model.train(True)
        writer.add_scalars('Loss&ACC/Valid', {
            'Loss': val_loss,
            'ACC': val_ACC
        }, global_step)
        writer.add_scalars(
            'Loss', {'Train': train_loss/batch, 'Valid': val_loss}, global_step)
        writer.add_scalars(
            'ACC', {'Train': train_corrects/(batch_size*batch), 'Valid': val_ACC}, global_step)
    writer.close()
    y, m, d = time.localtime()[:3]
    torch.save(model, f'{y}_{m}_{d}_res50.pth')


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    model = model()
    model.to(device=device)
    try:
        train_model(dirname=args.dir,
                    model=model,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), './INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

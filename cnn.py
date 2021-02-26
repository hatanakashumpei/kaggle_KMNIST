#!/usr/bin/env python
# coding: utf-8
"""
KMNISTデータセットによるクラス分類
CNNを用いて実装する
入力画像は 28*28の白黒画像で10クラス分類を行う
"""


# modules

import time
import os
import random

import numpy as np
from numpy.core.fromnumeric import amax
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from tqdm import tqdm

from IPython.display import FileLink

from torchvision.models import resnet18
# import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

# import EarlyStopping
import early_stopping_pytorch


def seed_everything(seed=42):
    """seedを固定させる
    [参考]
    https://qiita.com/si1242/items/d2f9195c08826d87d6ad

    Args:
        seed (int, optional): seedの設定値. Defaults to 42.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Setting Global Variable

INPUT_DIR = './'

PATH = {
    'train': os.path.join(INPUT_DIR, 'train.csv'),
    'sample_submission': os.path.join(INPUT_DIR, 'sample_submission.csv'),
    'train_image_dir': os.path.join(INPUT_DIR, 'train_images/train_images'),
    'test_image_dir': os.path.join(INPUT_DIR, 'test_images/test_images'),
}

ID = 'fname'
TARGET = 'label'

SEED = 42

# GPU settings for PyTorch (explained later...)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters for neural network. We will see the details later...
PARAMS = {
    'valid_size': 0.2,
    'batch_size': 64,
    'epochs': 50,
    'lr': 0.001,
    'valid_batch_size': 256,
    'test_batch_size': 256,
    'patience': 5,
    'n-fold': 5
}


class KMNISTDataset(Dataset):
    """Data Loaderの作成
    [参考]
    https://qiita.com/takurooo/items/e4c91c5d78059f92e76d

    Pytorchには他にもTransforms,DataLoaderがある
    Transforms:データの前処理
    DataLoader:データセットからデータをバッチサイズにかためて返すモジュール

    Args:
        Dataset : データとそれに対応するラベルを返すモジュール
    """
    def __init__(self, fname_list, label_list, image_dir, transform=None):
        super().__init__()
        self.fname_list = fname_list
        self.label_list = label_list
        self.image_dir = image_dir
        self.transform = transform

    # Datasetを実装するときにはtorch.utils.data.Datasetを継承する
    # __len__と__getitem__を実装する

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        fname = self.fname_list[idx]
        label = self.label_list[idx]

        image = cv2.imread(os.path.join(self.image_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            image = self.transform(image)
        # __getitem__でデータを返す前にtransformでデータに前処理をしてから返すことがポイント
        return image, label


class ResNetKMNIST(nn.Module):
    """ResNetのモデルを作成する
    [引用したサイト]
    https://github.com/marrrcin/pytorch-resnet-mnist

    """
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(
                                1,
                                64,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3),
                                bias=False
                            )
        self.loss = nn.CrossEntropyLoss()

    @auto_move_data
    def forward(self, x):
        return self.model(x)


def split_train_dataset(input_train_df):
    """訓練データをtrainとvalidに分割する

    Returns:
        df: train_df, valid_df
    """
    train_df, valid_df = train_test_split(
                            input_train_df, test_size=PARAMS['valid_size'],
                            random_state=SEED,
                            shuffle=True
                        )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    return train_df, valid_df


def make_train_dataset(train_df, valid_df):
    """train用のDatasetを作成する
    """
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    # numpy.arrayで読み込まれた画像をPyTorch用のTensorに変換します．
                    transforms.Normalize((0.5, ), (0.5, ))
                    # 正規化の処理も加えます。
                ])

    train_dataset = KMNISTDataset(
                        train_df[ID],
                        train_df[TARGET],
                        PATH['train_image_dir'],
                        transform=transform
                    )
    valid_dataset = KMNISTDataset(
                        valid_df[ID],
                        valid_df[TARGET],
                        PATH['train_image_dir'],
                        transform=transform
                    )

    # DataLoaderを用いてバッチサイズ分のデータを生成します。shuffleをtrueにすることでデータをshuffleしてくれます
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=PARAMS['batch_size'],
                        shuffle=True
                    )
    valid_dataloader = DataLoader(
                        valid_dataset,
                        batch_size=PARAMS['valid_batch_size'],
                        shuffle=False
                    )

    return train_dataloader, valid_dataloader


def make_kflod_train_dataset(train_df):
    """KFoldするためのtrainDatasetを作成する
    参考：https://naokiwifruit.com/2019/12/10/how-to-select-fold-cross-validation/
    """
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    # numpy.arrayで読み込まれた画像をPyTorch用のTensorに変換します．
                    transforms.Normalize((0.5, ), (0.5, ))
                    # 正規化の処理も加えます。
                ])
    outdataset = KMNISTDataset(
                        train_df[ID],
                        train_df[TARGET],
                        PATH['train_image_dir'],
                        transform=transform
            )

    return outdataset


def make_test_dataset():
    """テストデータのDatasetを作成する

    """
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, ), (0.5, ))
                ])

    test_dataset = KMNISTDataset(
                sample_submission_df[ID],
                sample_submission_df[TARGET],
                PATH['test_image_dir'],
                transform=transform
                )

    test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=PARAMS['test_batch_size'],
                    shuffle=False
                    )

    return test_dataloader


def accuracy_score_torch(y_pred, y):
    """予測と正解を入力してaccuracyを返す

    """
    y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
    y = y.cpu().numpy()

    return accuracy_score(y_pred, y)


def train(model, train_loader):
    """
    学習
    """
    model.train()
    train_loss_list = []
    train_accuracy_list = []

    for x, y in tqdm(train_loader):
        # 先ほど定義したdataloaderから画像とラベルのセットのdataを取得
        x = x.to(dtype=torch.float32, device=DEVICE)
        y = y.to(dtype=torch.long, device=DEVICE)
        # pytorchでは通常誤差逆伝播を行う前に毎回勾配をゼロにする必要がある
        optim.zero_grad()
        # 順伝播を行う
        y_pred = model(x)
        # lossの定義 今回はcross entropyを用います
        loss = criterion(y_pred, y)
        # 誤差逆伝播を行なってモデルを修正します(誤差逆伝播についてはhttp://hokuts.com/2016/05/29/bp1/)
        loss.backward()  # 逆伝播の計算
        # 逆伝播の結果からモデルを更新
        optim.step()

        train_loss_list.append(loss.item())
        train_accuracy_list.append(accuracy_score_torch(y_pred, y))

    return model, np.mean(train_loss_list), np.mean(train_accuracy_list)


def valid(model, valid_loader):
    """
    検証
    """
    model.eval()
    valid_loss_list = []
    valid_accuracy_list = []

    for x, y in tqdm(valid_loader):
        x = x.to(dtype=torch.float32, device=DEVICE)
        y = y.to(dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)

        valid_loss_list.append(loss.item())
        valid_accuracy_list.append(accuracy_score_torch(y_pred, y))

    return model, np.mean(valid_loss_list), np.mean(valid_accuracy_list)


def eval(models, test_loader):
    """
    testデータの予測
    """
    print("========evaluate========")
    all_predictions = np.zeros((len(sample_submission_df), 10))
    # それぞれのfoldで学習させたモデルで評価しその平均をとる
    for fold_idx in range(PARAMS['n-fold']):
        model = ResNetKMNIST().to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(models[fold_idx]))
        predictions = []

        for x, _ in test_loader:
            x = x.to(dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                y_pred = model(x)
                predictions.append(y_pred.cpu().numpy())
        
        predictions =  np.concatenate(predictions)
        all_predictions += predictions / len(models)

    # 提出データの作成
    all_predictions = np.argmax(all_predictions, axis=1).tolist()
    sample_submission_df[TARGET] = all_predictions

    sample_submission_df.to_csv('submission.csv', index=False)
    FileLink('submission.csv')

    # 予測結果を可視化
    sns.countplot(x=TARGET, data=sample_submission_df)
    plt.title('test prediction label distribution')
    plt.savefig('test_prediction_label_distribution.jpg')


if __name__ == '__main__':
    seed_everything(SEED)

    TrainDfBefore = pd.read_csv(PATH['train'])
    sample_submission_df = pd.read_csv(PATH['sample_submission'])

    print(f'number of train data: {len(TrainDfBefore)}')
    print(f'number of test data: {len(sample_submission_df)}')
    print(f'number of unique label: {TrainDfBefore[TARGET].nunique()}')

    # datasetの作成
    dataset = make_kflod_train_dataset(TrainDfBefore)

    fold = StratifiedKFold(n_splits=PARAMS['n-fold'], random_state=SEED, shuffle=True)

    cv = 0
    indexs = fold.split(TrainDfBefore[ID], TrainDfBefore[TARGET])

    models = []

    for fold_idx, (train_idx, valid_idx) in enumerate(indexs):
        print('======================================')
        print('fold {}'.format(fold_idx + 1))

        # modelの呼び出し
        model = ResNetKMNIST().to(DEVICE)
        # optimaizerは今回Adamを使用
        optim = Adam(model.parameters(), lr=PARAMS['lr'])
        # LambdaLRを用いて学習率を変化させる
        # [参考](https://katsura-jp.hatenablog.com/entry/2019/01/30/183501)
        scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 0.85 ** epoch)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, batch_size=PARAMS['batch_size'])
        valid_loader = DataLoader(Subset(dataset, valid_idx), shuffle=False, batch_size=PARAMS['valid_batch_size'])

        # initialize the early_stopping object
        # early stopping patience; how long to wait
        # after last time validation loss improved.
        early_stopping = early_stopping_pytorch.pytorchtools.EarlyStopping(
                        patience=PARAMS['patience'],
                        verbose=True
                        )

        for epoch_idx in range(PARAMS['epochs']):
            # epochループを回す
            start_time = time.time()

            model, train_loss, train_acc = train(model, train_loader)
            model, valid_loss, valid_acc = valid(model, valid_loader)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print(
                'Epoch: %d / %d' % (epoch_idx + 1, PARAMS['epochs']),
                " | time in %d minutes, %d seconds" % (mins, secs)
            )
            print('\tLoss: {:.4f}(train)\t|\tAcc: {:.1f}%(train)'.format(
                train_loss,
                train_acc * 100
            ))
            print('\tLoss: {:.4f}(valid)\t|\tAcc: {:.1f}%(valid)'.format(
                valid_loss,
                valid_acc * 100
            ))

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 学習率を更新
            scheduler.step()
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load('checkpoint.pt'))

        cv += valid_loss / fold.n_splits
        # save model for each fold index
        modelfilename = 'checkpoint_fold_{}.pt'.format(fold_idx + 1)
        print('save model for {}'.format(modelfilename))
        torch.save(model.state_dict(), modelfilename)
        models.append(modelfilename)

    print('cv {}'.format(cv))
    TestDataLoader = make_test_dataset()
    eval(models, TestDataLoader)

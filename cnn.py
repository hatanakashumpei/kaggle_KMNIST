#!/usr/bin/env python
# coding: utf-8
"""
KMNISTデータセットによるクラス分類
CNNを用いて実装する
入力画像は 28*28の白黒画像で10クラス分類を行う
"""


# modules

import os
import random

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from IPython.display import FileLink


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
    'epochs': 5,
    'lr': 0.001,
    'valid_batch_size': 256,
    'test_batch_size': 256,
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


class Net(nn.Module):
    """モデルを定義する
    ここではCNNで実装する
    """
    def __init__(self):
        super(Net, self).__init__()
        # 出力チャンネル数6, kernel size 5のCNNを定義する
        # 畳み込みの定義はPytorchの場合torch.nn.Conv2dで行います。ヒント:白黒画像とはチャネル数いくつかは自分で考えよう
        # 公式documentで使い方を確認する力をつけてほしいので、自分でconv2dなどの使い方は調べよう
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 出力チャネル数12, kernel_size 3のCNNを定義する 上記と同様に今度は自分で書いてみよう
        self.conv2 = nn.Conv2d(6, 12, 3)

        # Maxpoolingの定義(fowardでするのでもどっちでも)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # Linearの定義
        # 線形変換を行う層を定義してあげます: y = Wx + b
        # self.conv1, conv2のあと，maxpoolingを通すことで，
        # self.fc1に入力されるTensorの次元は何になっているか計算してみよう！
        # これを10クラス分類なので，10次元に変換するようなLinear層を定義します

        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        # forward関数の中では，，入力 x を順番にレイヤーに通していきます．みていきましょう
        # まずは，画像をCNNに通します
        x = self.conv1(x)

        # 活性化関数としてreluを使います
        x = F.relu(x)

        # 次に，MaxPoolingをかけます．
        x = self.maxpool(x)

        # 2つ目のConv層に通します
        x = self.conv2(x)

        # MaxPoolingをかけます
        x = self.maxpool(x)

        # 少しトリッキーなことが起きます．
        # CNNの出力結果を fully-connected layer に入力するために
        # 1次元のベクトルにしてやる必要があります
        # 正確には，　(batch_size, channel, height, width)
        # --> (batch_size, channel * height * width)
        x = x.view(batch_size, -1)

        # linearと活性化関数に通します
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = F.relu(x)
        return x


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


def main(train_dataloader, valid_dataloader, test_dataloader):
    """学習し、モデルの予測を行う
        ここはあとでクラスで書くべきである...
    """
    model = Net().to(DEVICE)

    optim = Adam(model.parameters(), lr=PARAMS['lr'])
    criterion = nn.CrossEntropyLoss()

    # 学習
    for epoch in range(PARAMS['epochs']):
        # epochループを回す
        model.train()
        train_loss_list = []
        train_accuracy_list = []

        for x, y in tqdm(train_dataloader):
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

        model.eval()
        valid_loss_list = []
        valid_accuracy_list = []

        for x, y in tqdm(valid_dataloader):
            x = x.to(dtype=torch.float32, device=DEVICE)
            y = y.to(dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)

            valid_loss_list.append(loss.item())
            valid_accuracy_list.append(accuracy_score_torch(y_pred, y))

        print(
            'epoch: {}/{} - loss: {:.5f} - accuracy: {:.3f} - val_loss: {:.5f} - val_accuracy: {:.3f}'.format(
                epoch,
                PARAMS['epochs'],
                np.mean(train_loss_list),
                np.mean(train_accuracy_list),
                np.mean(valid_loss_list),
                np.mean(valid_accuracy_list)
            ))

    # モデルの予測
    model.eval()
    predictions = []

    for x, _ in test_dataloader:
        x = x.to(dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
            y_pred = y_pred.tolist()

        predictions += y_pred

    # 提出データの作成
    sample_submission_df[TARGET] = predictions

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

    TrainDf, ValidDf = split_train_dataset(TrainDfBefore)
    TrainDataLoader, ValidDataLoader = make_train_dataset(TrainDf, ValidDf)
    TestDataLoader = make_test_dataset()

    main(TrainDataLoader, ValidDataLoader, TestDataLoader)

#!/usr/bin/env python
# coding: utf-8

# とりあえずこのnotebookでpytorchに慣れましょう！！！
# 
# わからないことがあったらどんどん質問してくださいね
# 
# まずは上から順に実行してみましょう

# # Input Modules

# In[1]:


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
from torch.optim import SGD,Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

#get_ipython().run_line_magic('matplotlib', 'inline')







# seedを固定しないとモデルの精度が毎回異なることがある。
# 比較がしずらい場合がある。
# そのため、どのモデルがいいのかを比較するためにSeedは固定している

# In[2]:


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# データセットファイルの定義とハイパパラメータの定義を行います
# 
# ニューラルネットワークには学習可能なパラメータ(重みやバイアスなど）と自分で値を決める必要があるハイパパラメータが存在します。
# 
# ハイパパラメータを以下に定義します。

# In[3]:


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
seed_everything(SEED)

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


# validation data
# trainで学習したデータを確認する
# testデータは未知のものを評価するもの

# train_dfにtrain用データを入れます

# In[4]:


train_df = pd.read_csv(PATH['train'])
sample_submission_df = pd.read_csv(PATH['sample_submission'])


# trainデータの大きさを確認します
# 
# kaggleをやる際には（研究やる上でも？）データセットのサイズなどの確認は重要なのでマストでやれると良いですね

# In[5]:


print(f'number of train data: {len(train_df)}')
print(f'number of test data: {len(sample_submission_df)}')


# 続いて今回は１０クラス分類なのでラベル数の確認もします
# テストデータのクラス分類の精度をもとにコンペを競う

# In[6]:


print(f'number of unique label: {train_df[TARGET].nunique()}')


# データセットのラベル数を可視化して見てみましょう。全て同じ枚数で偏りがないデータセットであることが分かりますね。

# In[7]:

"""
#sns.countplot(train_df[TARGET])
#print(train_df)
fig_train = plt.figure()
fig1_train = sns.countplot(x=TARGET,data=train_df)
fig1_train.suptitle('train label distribution')
# plt.show()
fig1_train.savefig('train_label_distribution.jpg')
"""
# trainデータの中身を確認します
# 

# In[8]:


train_df.head()


# 試しにtrainデータの内容を可視化してみましょう

# In[9]:


sample = train_df.groupby(TARGET).first().reset_index()

#fig, ax = plt.subplots(2, 5)
#fig.set_size_inches(4 * 5, 4 * 2)

for i, row in sample.iterrows():
    fname, label = row[ID], row[TARGET]
    img = cv2.imread(os.path.join(PATH['train_image_dir'], fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ax[i//5,i%5].imshow(img, 'gray')
    #ax[i//5,i%5].set_title(f'{fname} - label: {label}')


# 入力データの画像サイズの確認をします。
# 
# 入力サイズはモデルを作る上で必須なので確認しておきましょう

# In[10]:


print(f'shape of image: {img.shape}')


# # Data Loaderの作成

# pytorchではdataのロードと前処理を行うことができるものが存在します。
# 
# ここではざっくりとした説明するので初めましての人はほえええそうなんだくらいで聞いといてください。
# 
# 以下のqiita記事なども分かりやすいのでみて勉強するのもおすすめです。
# https://qiita.com/takurooo/items/e4c91c5d78059f92e76d
# 

# Pytorchにはtransforms, Dataset, Dataloaderという三種の神器があってこれを使いこなせるようになるとデータロード周りが非常に楽にスマートになります． (多分tensorflowにはない？）

# * Transforms: データの前処理を担当
# 
# * Dataset: データとそれに対応するラベルを返すモジュール(transformsで前処理したものを返す）
# 
# * DataLoader: データセットからデータをバッチサイズに固めて返すモジュール

# In[11]:


class KMNISTDataset(Dataset):
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


# # Modelクラスの定義

# 自分で学習させたいニューラルネットワークをここで組みます。
# 
# 基本的にPyTorchでは，モデルをクラスとして宣言します．
# PyTorchを使うときは，nn.Moduleを継承したクラスを宣言して．その中にforward関数を作ります．
# 
# 使いたいレイヤー（CNNとか，Denseとか，activation function）を__init__のなかで定義して，
# forwardの中では，ニューラルネットに行わせたい処理を記載していきます．
# 
# 1. __init__(self): クラスからインスタンスを作成したときに自動的に実行される関数です．
# PyTorchでは基本的にここでモデルの中身を設定してやります．
# 基本的には「学習しないといけない重み」を持っている層がここで設定されます．
# 他にも，活性化関数などをここで定義することもできます
# 
# 1. forward(self, x) (順伝播関数)
# PyTorchでは，実際に何かをニューラルネットワークに入力した場合にはこの関数の中身が実行され，計算結果が出力されます．
# 
# ものすごーい簡略化してかくと，こんな感じになります．
# 
# model = Net() <-- クラスのインスタンスを作成する
# 
# output = model(input_images) <-- これを実行すると，
# input_images が forward()に渡されて，計算結果が output に渡される

# In[12]:


# 入力は28*28の白黒画像で10クラス分類を行う

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):
        super().__init__()
        # nn.Linearは fully-connected layer (全結合層)のことです．
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # 1次元のベクトルにする
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x


# In[13]:


# 以下を埋めてみよう
# 今回の研修では
# モデルとして入力から出力チャネル数6, kernel_size5の畳み込み層→Maxpooling(2×2)→出力チャネル数12, kernel_size3の畳み込み層
# → MaxPooling(2×2)→1次元にする→Linearで10次元出力
# というモデルを作成してください(strideなどは考えないでください)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 出力チャンネル数6, kernel size 5のCNNを定義する
        # 畳み込みの定義はPytorchの場合torch.nn.Conv2dで行います。ヒント:白黒画像とはチャネル数いくつかは自分で考えよう
        # 公式documentで使い方を確認する力をつけてほしいので、自分でconv2dなどの使い方は調べよう
        self.conv1 = nn.Conv2d(1,6,5)
        # 出力チャネル数12, kernel_size 3のCNNを定義する 上記と同様に今度は自分で書いてみよう
        self.conv2 = nn.Conv2d(6,12,3)
        
        # Maxpoolingの定義(fowardでするのでもどっちでも)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        
        # Linearの定義
        # 線形変換を行う層を定義してあげます: y = Wx + b
        # self.conv1, conv2のあと，maxpoolingを通すことで，
        # self.fc1に入力されるTensorの次元は何になっているか計算してみよう！
        # これを10クラス分類なので，10次元に変換するようなLinear層を定義します
        
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        

    
    def forward(self, x):
        batch_size = x.shape[0]
        # forward関数の中では，，入力 x を順番にレイヤーに通していきます．みていきましょう．    
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
        # 正確には，　(batch_size, channel, height, width) --> (batch_size, channel * height * width)
        x = x.view(batch_size, -1)
        
        # linearと活性化関数に通します
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        x = F.relu(x)
        return x


# ここが埋めれたら下記で model = MLP().to(DEVICE)になってる部分のMLPをNet()に書き直してsubmissionしてみましょう!　１回目より精度は上がったかな？

# In[14]:


net = Net()


# trainデータをtrainとvalidに分割します。pythonではvalid_sizeなどを指定することでランダムにtrainとvalidへの分割ができます。
# 
# ここで質問trainデータ, validデータ, testデータってなんでしたっけ？

# In[15]:


train_df, valid_df = train_test_split(
    train_df, test_size=PARAMS['valid_size'], random_state=SEED, shuffle=True
)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)


# 実際にDatasetを作成していきます。
# transformsはデータの前処理をするものでしたね。

# In[16]:


transform = transforms.Compose([
    transforms.ToTensor(),
    # numpy.arrayで読み込まれた画像をPyTorch用のTensorに変換します．
    transforms.Normalize((0.5, ), (0.5, ))
    #正規化の処理も加えます。
])

train_dataset = KMNISTDataset(train_df[ID], train_df[TARGET], PATH['train_image_dir'], transform=transform)
valid_dataset = KMNISTDataset(valid_df[ID], valid_df[TARGET], PATH['train_image_dir'], transform=transform)

# DataLoaderを用いてバッチサイズ分のデータを生成します。shuffleをtrueにすることでデータをshuffleしてくれます
train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=PARAMS['valid_batch_size'], shuffle=False)


# PyTorchではGPUを使うために以下のコードを実行します。GPUに明示的に送る必要があるんですね。

# In[17]:


#model = MLP().to(DEVICE)
model = Net().to(DEVICE)


# In[18]:


# model = model.to("cuda")
# tensor = tensor.to("cuda")


# In[19]:


optim = SGD(model.parameters(), lr=PARAMS['lr'])
criterion = nn.CrossEntropyLoss()


# optimaizerを設定すると精度があがるらしい

# 予測と正解を入力してaccuracyを返す関数を定義します
# 

# In[20]:


def accuracy_score_torch(y_pred, y):
    y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
    y = y.cpu().numpy()

    return accuracy_score(y_pred, y)


# 早速学習を回してみましょう
# 
# 

# In[21]:


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
        loss.backward() # 逆伝播の計算
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
    
    print('epoch: {}/{} - loss: {:.5f} - accuracy: {:.3f} - val_loss: {:.5f} - val_accuracy: {:.3f}'.format(
        epoch,
        PARAMS['epochs'], 
        np.mean(train_loss_list),
        np.mean(train_accuracy_list),
        np.mean(valid_loss_list),
        np.mean(valid_accuracy_list)
    ))


# テストデータセットも同様にtransformなどを施しておきます

# In[22]:


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

test_dataloader = DataLoader(test_dataset, batch_size=PARAMS['test_batch_size'], shuffle=False)


# 実際にモデルの予測を行います

# In[23]:


model.eval()
predictions = []

for x, _ in test_dataloader:
    x = x.to(dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
        y_pred = y_pred.tolist()
        
    predictions += y_pred


# In[24]:


sample_submission_df[TARGET] = predictions
#print(sample_submission_df)

# kaggle提出用に以下のようなcsvを作成しましょう

# In[25]:


sample_submission_df.to_csv('submission.csv', index=False)
from IPython.display import FileLink
FileLink('submission.csv')


# 予測結果を確認します

# In[26]:

#sns.countplot(sample_submission_df[TARGET])
sns.countplot(x=TARGET,data=sample_submission_df)
plt.title('test prediction label distribution')
# plt.show()
plt.savefig('test_prediction_label_distribution.jpg')


# In[27]:

"""
fig, ax = plt.subplots(2, 5)
fig.set_size_inches(4 * 5, 4 * 2)

for i, row in sample_submission_df.iloc[:10,:].iterrows():
    fname, label = row[ID], row[TARGET]
    img = cv2.imread(os.path.join(PATH['test_image_dir'], fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax[i//5,i%5].imshow(img, 'gray')
    ax[i//5,i%5].set_title(f'{fname} - label: {label}')


# ここまで実行できたら次は自分でCNNのモデルを作成して動かしてみましょう。
"""
# In[ ]:





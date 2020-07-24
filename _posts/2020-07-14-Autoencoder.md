---
layout: post
title: Autoencoder
subtitle : 
tags: [Python]
author: Yeabin
comments : True
---



# 오토 인코더(Autoencoder)

* 오토 인코더는 출력이 입력 데이터와 같아지도록 학습한 네트워크이다.

* 오토 인코더의 경우 별도의 Supervised learning용 label 이나 출력값이 없으므로 Unsupervised learning에 해당한다.(slef-supervised learning이라고도 한다.)

* 단층 혹은 복층 구조로 구성할 수 있으며, 차원축소, 잡음제거, 이상 데이터 검출, 사전학습 등에 활용될 수 있다.

* 왼쪽의 경우 단층구조(Hidden layer가 1개), 오른쪽의 경우 복층구조(Hidden layer가 여러개)이다.

  ![autoencoder image]({{ site.baseurl }}/assets/img/autoencoder-1594706468099.PNG)

  * 데이터를 압축/확장 하는 과정을 Encoder, 데이터를 복원하는 과정을 Decoder라고 한다.

## mnist 데이터를 Autoencoder를 이용해 차원을 축소해보자

* 28x28 = 769의 mnist 데이터를 10x10=100 이미지로 축소하고, K-Means로 10개의 군집으로 분류하자.
* 우선, mnist 데이터를 다운로드하고 저장한다.(이후 데이터 사용할 경우 데이터를 읽어오면 된다.)

~~~
from sklearn.datasets import fetch_openml
import pickle

mnist = fetch_openml('mnist_784')
with open('dataset/mnist.pickle','wb') as f:
	pickle.dump(mnist,f)
~~~

~~~
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

with open('dataset/mnist.pickle', 'rb') as f:
        mnist = pickle.load(f) # 저장된 mnist 데이터 읽어오기
~~~

~~~
inputX = mnist.data[:3000, :]
imageX = inputX.copy()
inputY = mnist.target[:3000] # inputY는 학습과는 무관, 나중에 결과 확인을 위한 데이터
~~~

현재 inputX의 shape은 아래 그림과 같다.

![shape image]({{ site.baseurl }}/assets/img/IputX_shape.PNG)

그림에서 가로 방향으로 표준화를 하기 위해 표준화 전후로  transpose 한다.

~~~
sc = StandardScaler()
inputX = sc.fit_transform(inputX.T).T
~~~

~~~
nInput = inputX.shape[1]
nFeature = 100 
nOutput = nInput

xInput = Input(batch_shape=(None, nInput))
xEncoder = Dense(256, activation='relu')(xInput)
xEncoder = Dense(nFeature, activation='relu')(xEncoder)
yDecoder = Dense(256, activation='relu')(xEncoder)
yDecoder = Dense(nOutput, activation='linear')(yDecoder)
model = Model(xInput, yDecoder)
encoder = Model(xInput, xEncoder)
model.compile(loss='mse', optimizer=Adam(lr=0.01))

# autoencoder를 학습
hist = model.fit(inputX, inputX, epochs=500, batch_size=100)
~~~

위의 코드는 아래의 그림과 같은 과정이다.

![model image]({{ site.baseurl }}/assets/img/auto_model.PNG)

~~~
# 학습된 autoencoder를 이용하여 입력 데이터의 차원을 축소
inputXE = encoder.predict(inputX)

# K-means++ 알고리즘으로 차원이 축소된 이미지를 10 그룹으로 분류
km = KMeans(n_clusters=10, init='k-means++', n_init=3, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(inputXE)
clust = km.predict(inputXE)
# cluster 별로 이미지 확인
f = plt.figure(figsize=(8, 2))
for k in np.unique(clust):
    # cluster가 i인 imageX image 10개를 찾는다.
    idx = np.where(clust == k)[0][:10]
    
    f = plt.figure(figsize=(8, 2))
    for i in range(10):
        image = imageX[idx[i]].reshape(28,28)
        ax = f.add_subplot(1, 10, i + 1)
        ax.imshow(image, cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(k)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
~~~

* 결과

![result image]({{ site.baseurl }}/assets/img/auto_output.PNG)


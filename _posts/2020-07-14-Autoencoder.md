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
* 
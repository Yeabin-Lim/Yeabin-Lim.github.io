---
layout: post
title: Generative Adversarial Nets
subtitle : 
tags: [Python]
author: Yeabin
comments : True
---



# Generative Adversarial Nets(GAN)

* GAN 은 Unsupervised Learning 방식으로 이미지, 문서, 음성 등의 데이터를 생성하는 알고리즘이다.

* GAN은 Discriminator Network 와 Generator Network로 구성되어 있다. 

  * Generator N/W의 경우 랜덤 데이터(Noise)를 입력 받아 실제 데이터(Real Data)와 유사한 가짜 데이터를 생성하고, Discriminator N/W의 경우 Real Data와 Fake Data를 구별한다.

    ![GAN_algorithm]({{ site.baseurl }}/assets/img/GAN_algorithm.PNG)

* Discriminator의 경우 0과 1 사이의 값(sigmoid)를 출력한다. 

  * Real Data가 들어갈 경우 1에 가까운 값이 출력되고, Fake Data가 들어갈 경우 0에 가까운 값이 출력된다.
  * Discriminator 과 Generator가 서로 균형(Nash Equilibrium) 상태에 도달하면 Discriminator는 Real Data와 Fake Data를 잘 구별하지 못하므로 0.5에 가까운 값을 출력한다.

* GAN의 Loss function

  ![GAN_Loss_function]({{ site.baseurl }}/assets/img/GAN_Loss_function.PNG)

  * Discriminator의 경우 max V(D,G)로 학습한다. D(x)=1 이고, D(G(z))=0 일 경우 최대

  * Generator의 경우 min V(G)로 학습한다. D(G(z))=1 일 경우 최소

    * Generator의 경우 logD(x)는 상수이므로 log(1-lD(G(z)))항만 신경쓰면 된다.

      ![GAN_generator_1]({{ site.baseurl }}/assets/img/GAN_generator_1.PNG)

      ![GAN_generator_2]({{ site.baseurl }}/assets/img/GAN_generator_2.PNG)

      * 두 함수 중 위의 함수보다 아래 함수를 이용할 경우 초기 학습 효과도 개선된다.

* GAN의 학습 과정

  ![GAN_learning]({{ site.baseurl }}/assets/img/GAN_learning.PNG)

  * 검정색 그래프: x의 분포, 초록색 그래프: G(z)의 분포
  * 학습이 진행되면서 G(z)의 분포가 x의 분포와 유사해지는 것을 확인할 수 있다.

참고: [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
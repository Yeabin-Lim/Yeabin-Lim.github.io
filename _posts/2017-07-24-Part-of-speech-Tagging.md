---
layout: post
title: Part-of-speech Tagging
subtitle : HMM
tags: [Python]
author: Yeabin
comments : True
---

# 품사 태깅(Part-of-speech Tagging)

* 문장에 사용된 단어들에 알맞은 품사를 결정하는 것을 "품사 태깅"이라 한다.
* 분석할 문장의 올바른 품사를 결정하기 위해서는 사전에 올바른 품사가 정의된 문서 `코퍼스`가 있어야 한다. 사전에 품사가 정의된 코퍼스를 `Tagged Corpus`라 하고, Tagged Corpus를 학습하여 분석할 문장의 품사를 결정한다.
  * Tagged Corpus의 학습은 Supervised learning에 속한다.
* 품사 태깅의 목적은 `문장의 의미 파악`과 `문법에 맞는 문장 생성`이다. 
  * "I write you"와 같이 의미가 통하지 않는 문장을 `word sallad`라 한다.
* 기계 번역과 챗봇의 경우 문장의 의미 파악과 구조에 맞게 문장 생성을 하기 때문에 품사 태깅이 필요하다.

### POS tagging Code

~~~
import nltk

text = "Today I learned about part of speech tagging."
token = nltk.word_tokenize(text)
nltk.pos_tag(token)
~~~

```
[('Today', 'NN'),
 ('I', 'PRP'),
 ('learned', 'VBD'),
 ('about', 'IN'),
 ('part', 'NN'),
 ('of', 'IN'),
 ('speech', 'NN'),
 ('tagging', 'NN'),
 ('.', '.')]
```

* `nltk.pos_tag()`를 사용해서 임의의 문장에 사용된 단어들의 품사를 파악할 수 있다.
* 품사 태깅의 과정
  ![POS image]({{ site.baseurl }}/assets/img/tagging_1.PNG)
* 품사 태깅의 경우 Corpus마다 차이가 있다.
  * 많이 사용하는 학습용 데이터로는 Penn Treebank corpus, Brown corpus, NPS Char corpus 등이 있다.



## 히든 마코프 모델(HMM)

### Markov Chain

* 현재 상태는 바로 직전 데이터에만 의존한다.(1차 Markov Chain)
* 현재 상태는 전전 상태에만 의존한다.(2차 Markov Chain)

### Hidden Markov Model(HMM)

* 관측 데이터에 직접 나타나지 않는 `Hidden State`가 존재한다.
* 관측 데이터는 동일 시점의 히든 상태에만 의존한다.
* 히든 상태는 이전 상태에 의존한다.

![HMM image]({{ site.baseurl }}/assets/img/HMM.PNG)

위 그림에서 X는 관측 데이터, Z는 히든 상태를 의미한다.

주식으로 예를 들어보면 X는 주가, 수익률, 거래량, 변동성 등을 의미한다. 오늘의 주가는 어제의 주가에 직접 의존하는 것이 아니라, 오늘의 상태에 의존하고, 오늘의 상태는 어제의 상태에 의존한다.

* HMM 3가지 관심 사항

![HMM image]({{ site.baseurl }}/assets/img/HMM_el.PNG)

1. 초기상태, 천이확률, 출력확률이 주어진 경우 관측치X 시퀀스가 나올 확률 계산 -> Forward, Backward 알고리즘
2. 초기상태, 천이확률, 출력확률이 주어진 상태에서 가장 가능성 있는 Z의 시퀀스를 추정함 -> Viterbi decoding 알고리즘
3. X 만 주어진 경우 초기상태, 천이확률, 출력확률 모두를 추론하고, Z의 시퀀스를 찾아내는 알고리즘 -> Baum-Welch 알고리즘

* Forward 알고리즘
  * 위 그림에서의 초기확률, 천이확률, 출력확률을 사용할 경우 `어제 걸었고 오늘 청소를 할 경우`의 확률을 구해보자.
  * 우선, 조건부 확률을 사용해 어제 `walk`라는 행동이 발생했을 확률을 계산하자.
    P('walk') = P('walk'|Rainy) + P('walk'|Sunny)
    = 0.6\*0.1+0.4\*0.6 = 0.30 이다.
  * 어제 `walk` 이란 행동이 발생하고 오늘 `clean`이란 행동이 발생할 경우의 수를 구해보자.
    1. Sunny - walk -> Sunny - clean
    2. Sunny - walk -> Rainy - clean
    3. Rainy - walk -> Sunny -> clean
    4. Rainy - walk -> Rainy -> clean
  * 1번 경우에 대한 확률을 구하면 다음과 같다.
    (어제 맑았을 확률)0.4* (맑았다는 가정에서 걸었을 확률)0.6 * (어제 맑았고 오늘 맑을 확률)0.6* (오늘이 맑았다는 가정에서 청소할 확률)0.1 = 0.0144
  * 위와 같이 모든 식을 계산할 경우 각각 0.0144, 0.048, 0.0018, 0.021이 나오고, `어제 걸었고 오늘 청소를 할 경우의 확률 =  0.0144+0.048+0.0018+0.021 = 0.0852`가 된다.

~~~
import numpy as np
from hmmlearn import hmm

# 히든 상태 정의
states = ["Rainy", "Sunny"]
nState = len(states)

# 관측 데이터 정의
observations = ["Walk", "Shop", "Clean"]
nObervation = len(observations)

# HMM 모델 빌드
model = hmm.MultinomialHMM(n_components=nState)
model.startprob_ = np.array([0.6, 0.4])
# pi
model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

# 관측 데이터 (Observations)
X = np.array([[0,2]]).T
# Forwad(/Backward) algorithm으로 x가 관측될 likely probability 계산
logL = model.score(X)
p = np.exp(logL)
print("\nProbability of [Walk, Clean] = %.4f%s" % (p*100, '%'))
~~~

위의 코드 결과를 확인해 보면 `Probability of [Walk, Clean] = 8.5200%` 로 위에서 계산한 확률과 같음을 확인할 수 있다.

* Viterbi 알고리즘
  * Viterbi 알고리즘의 경우 `walk -> clean -> shop -> walk`시퀀스가 관찰됐을 때 가장 가능한 Z-시퀀스를 찾는다.
  * Forward 알고리즘과 같은 형태로 행동에 대한 모든 확률을 구한 후 가장 확률이 높은 시퀀스를 찾는다.
  * Forward 예시를 기준으로 살펴보면, `walk -> clean`이라는 시퀀스에 대해 확률이 가장 높은 `Sunny -> Rainy`가 Z가 될 것이다.

~~~
import numpy as np
from hmmlearn import hmm

# 히든 상태 정의
states = ["Rainy", "Sunny"]
nState = len(states)

# 관측 데이터 정의
observations = ["Walk", "Shop", "Clean"]
nObervation = len(observations)

# HMM 모델 빌드
model = hmm.MultinomialHMM(n_components=nState)
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

# 관측 데이터 (Observations)
X = np.array([[0, 2]]).T

# Viterbi 알고리즘으로 히든 상태 시퀀스 추정 (Decode)
logprob, Z = model.decode(X, algorithm="viterbi")

# 결과 출력
print("\n  Obervation Sequence :", ", ".join(map(lambda x: observations[int(x)], X)))
print("Hidden State Sequence :", ", ".join(map(lambda x: states[int(x)], Z)))
print("Probability = %.6f" % np.exp(logprob))
~~~

위의 코드 결과를 확인해보면 위에서 계산한 확률과 같음을 확인할 수 있다.

```
Obervation Sequence : Walk, Clean
Hidden State Sequence : Sunny, Rainy
Probability = 0.048000
```

* Baum Welch 알고리즘
  * 관측값인 X만 가지고 가장 가능성이 높은 초기확률, 천이확률, 출력확률, Z의 시퀀스를 추정한다.
  * 위 변수에 대한 초깃값을 임의로 설정한 후 관측값인 X를 이용하여 E-M알고리즘을 반복하며 변수를 업데이트한다.  
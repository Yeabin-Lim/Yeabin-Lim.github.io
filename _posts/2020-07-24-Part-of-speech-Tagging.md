---
layout: post
title: Part of speech Tagging
subtitle : Corpus, NLTK
tags: [Python]
author: Yeabin
comments : True
---

# NLTK 설치

* 아나콘다 프롬프트에서 명령어 수행
  `> conda install -c anaconda nltk (or conda install nltk)`

* 데이터 다운로드: 

  > import nltk
  > nltk.download()

![nltk image]({{ site.baseurl }}/assets/img/nltk_down.PNG)

* `book` 선택 후 다운로드 실행

# Corpus

* `corpus(말뭉치)`는 자연어 분석 작업을 위해 만든 학습용 기초 데이터(문서)이다.
* [Gutenberg Corpus](http://www.gutenberg.org/)에서 약 25,000권의 영문 소설을 무료로 제공하고 있다.
* NLTK 패키지에서도 일부 제공하고 있으며, NLTK에 없는 데이터는 사이트에서 직접 받을 수 있다.

# Token

* 자연어 문서 분석을 위해 코퍼스 데이터를 필요에 맞게 작은 단위로 나누는 전처리 작업을 해야 한다. 이 작업을 토큰화(tokenization)라고 말하며, 단위를 토큰(token)이라 한다. 
* 토큰은 의미를 갖는 단어를 의미하며, 여기서의 단어는 단어 단위 외에도 단어구, 의미를 갖는 문자열로 간주되기도 한다.
* 임의의 문장을 단어 기준으로 토큰화 해보자.

~~~
import nltk
text = 'Today I learned about part of speech tagging.'
token = nltk.word_tokenize(text)
print(token)
~~~

```
['Today', 'I', 'learned', 'about', 'part', 'of', 'speech', 'tagging', '.']
```



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
  
  * Penn Treebank corpus에 사용된 품사 목록
  ![pos image]({{ site.baseurl }}/assets/img/pos_list.PNG)
    
    [출처](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

### 히든 마코프 모델과 POS-tagging

\* [히든 마코프 게시글](https://yeabin-lim.github.io/2020-07-24-Hidden-Markov-Model.html)

* `They can fish.`와 같은 문장에서 HMM 모델을 사용하여 pos tagging이 되는 과정을 살펴보자.
![poshmm image]({{ site.baseurl }}/assets/img/pos_hmm.PNG)
  
* HMM의 전이확률(transition probability)와 출력확률(Emission probability)를 이용하여 POS tagging을 한다.

* 문장이 길면 조합이 많아지므로 `Viterbi` 알고리즘을 사용해 가장 확률이 높은 tag 시퀀스로 품사를 태깅한다.
  
  * P(t = N,V,N| w = they can fish),P(t = N,V,V| w = they can fish), P(t = V,V,N| w = they can fish) 등의 확률을 계산하여 `they`, `can`, `fish` 에 품사를 태깅한다.
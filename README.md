# AIX-DeepLearning

## 기말대체 프로젝트

# Title: 음식 이미지 인식을 통한 칼로리 계산

# Introduction

## Background & Scheme
CNN 기법을 이용하여 서로 다른 이미지를 종류에 따라 분류하는 방법은 알려져 있다. 이를 기반으로 하여 모델이 인터넷에서 추출한 데이터셋을 바탕으로 학습한 결과를 근거로 처음 접하는 한식의 이미지를 보고 이미지에 해당하는 음식의 단위 제공량 당 열량을 추측하게끔 하고자 한다. Google image searching을 통해 얻은 중식과 일식의 검색량 상위에 해당하는 각 10종류의 음식 데이터셋에서 음식 종류별로 실제 단위 제공량 당 열량을 부여하고, 모델을 학습시킬 계획이다. 학습을 마친 모델이 주어진 한식의 이미지가 경험한 이미지 중 확률적으로 어떤 것에 가까운지 판단하여 해당 이미지가 indicating하는 음식의 단위 제공량 당 열량의 추정치를 결과로 내놓는 일련의 과정을 Tensorflow 내의 API, Keras를 이용한 CNN으로 구현하겠다.

## Challenges
본 프로젝트에서는 음식(한식) 데이터에 대하여 이미지 속 음식의 양을 고려하지 않고 음식의 종류만을 추정하여 해당 음식의 열량을 결괏값으로 내놓는 모델을 설계할 것이다. 이미지에 담긴 음식의 양(개수를 셀 수 있는 음식의 경우 그 개수)을 따지는 과정을 거쳐 이미지 속 모든 음식의 열량을 추정하는 모델을 설계하는 것은 도전적인 과제이다. 이미지가 나타내는 음식이 가지는 외관 상의 속성(열량에 초점을 맞추어)을 넘어 그 총량을 알 수 있다면 주어지는 이미지의 열량에 대해 더 높은 수준으로 대답할 수 있다.

# Our Method

## Dataset
Python의 Selenium module을 활용하여 Google에서 image crawling을 통해 Google 검색량 상위 항목에 해당하는 음식 중 모델 학습에 효과적일 것으로 판단되는 10종류의 일식(nabe, soba, ramen, Japanese curry, sushi, udon, karaage, onigiri, gyudon, okonomiyaki)와 10종류의 중식(congee, dong po rou, baozi, chaofan, zhajiangmian, sweet and sour pork, mapotofu, wonton soup, mooncake, pecking duck)의 이미지를 각 종류별로 약 400개씩 수집한다. 이후 수집한 이미지 중 서로 중복되는 이미지를 제거하여 트레이닝을 위한 데이터셋을 얻는다. 검색량 상위를 차지하는 한식에 대해서도 같은 방식의 과정을 수행하여 모델이 열량을 추정하는 데 사용할 한식 이미지 데이터셋을 얻는다.

## Image Crawling
학습을 위한 이미지를 수집하기 위한 Image Crawling 과정은 다음과 같다. Chrome driver를 이용해서 Google image searching을 위한 URL을 탐색하는 driver를 생성한다. Dictionary 형식으로 저장한 일식 및 중식의 종류에 따른 검색어 각각에 대해 image searching을 순차적으로 시행한다. 각 시행에서 중복되는 이미지를 제외한 개별적인 이미지가 존재하는 URL을 얻고 모든 URL에서 이미지를 다운로드하여 Dictionary에 저장된 음식의 이미지 묶음을 생성한다.

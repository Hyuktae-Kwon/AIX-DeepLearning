# AIX-DeepLearning

## 기말대체 프로젝트

# Title: 음식 이미지 인식을 통한 칼로리 계산

# Introduction
CNN 기법을 이용하여 서로 다른 이미지를 종류에 따라 분류하는 방법은 알려져 있다. 이를 기반으로 하여 모델이 인터넷에서 추출한 데이터셋을 바탕으로 학습한 결과를 근거로 처음 접하는 한식의 이미지를 보고 이미지에 해당하는 음식의 단위 제공량 당 열량을 추측하게끔 하고자 한다. Google image searching을 통해 얻은 중식과 일식의 검색량 상위에 해당하는 각 10종류의 음식 데이터셋에서 음식 종류별로 실제 단위 제공량 당 열량을 부여하고, 모델을 학습시킬 계획이다. 학습을 마친 모델이 주어진 한식의 이미지가 경험한 이미지 중 확률적으로 어떤 것에 가까운지 판단하여 해당 이미지가 indicating하는 음식의 단위 제공량 당 열량의 추정치를 결과로 내놓는 일련의 과정을 Tensorflow 내의 API, Keras를 이용한 CNN으로 구현하겠다.

# Our Method
-Dataset
Python의 Selenium module을 활용하여 Google에서 image crawling을 통해 Google 검색량 상위 항목에 해당하는 음식 중 모델 학습에 효과적일 것으로 판단되는 10종류의 일식(nabe, soba, ramen, Japanese curry, sushi, udon, karaage, onigiri, gyudon, okonomiyaki)와 10종류의 중식(congee, dong po rou, baozi, chaofan, zhajiangmian, sweet and sour pork, mapotofu, wonton soup, mooncake, pecking duck)의 이미지를 각 종류별로 약 400개씩 수집한다.++++

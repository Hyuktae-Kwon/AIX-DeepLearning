# AIX-DeepLearning

## 기말대체 프로젝트

# Title: 음식 이미지 인식을 통한 칼로리 계산

# Introduction

## Background & Scheme
CNN 기법을 이용하여 서로 다른 이미지를 종류에 따라 분류하는 방법은 알려져 있다. 이를 기반으로 하여 모델이 인터넷에서 추출한 데이터셋을 바탕으로 학습한 결과를 근거로 처음 접하는 한식의 이미지를 보고 이미지에 해당하는 음식의 단위 제공량 당 열량을 추측하게끔 하고자 한다. Google image searching을 통해 얻은 중식과 일식의 검색량 상위에 해당하는 각 10종류의 음식 데이터셋에서 음식 종류별로 실제 단위 제공량 당 열량을 부여하고, 모델을 학습시킬 계획이다. 학습을 마친 모델이 주어진 한식의 이미지가 경험한 이미지 중 확률적으로 어떤 것에 가까운지 판단하여 해당 이미지가 indicating하는 음식의 단위 제공량 당 열량의 추정치를 결과로 내놓는 일련의 과정을 Tensorflow 내의 API, Keras를 이용한 CNN으로 구현하겠다.

## Proposal

본 프로젝트에서의 모델과 데이터셋을 이용해 처음 접하는 이미지가 이미 경험적으로 알고 있는 이미지 중 어떤 이미지와 유사한지 확률적인 근거를 가지고 판단했을 때 실제로 그 판단의 결과가 가지는 정확도에 대해 알아볼 수 있다. 동아시아 문화권에 속하여 대한민국과 일부 유사한 식문화를 가지는 일본과 중국의 음식 이미지를 학습시킨다. 학습 후 한식의 이미지를 접했을 때의 결과를 통해 새로운 이미지(여기서는 한식의)와 높은 관련도를 가지는 범주에 있는 음식에 대한 학습이 음식 열량의 추정에 유의미한 도움을 주는지 알고자 한다.

## Challenges
본 프로젝트에서는 음식(한식) 데이터에 대하여 이미지 속 음식의 양을 고려하지 않고 음식의 종류만을 추정하여 해당 음식의 열량을 결괏값으로 내놓는 모델을 설계할 것이다. 이미지에 담긴 음식의 양(개수를 셀 수 있는 음식의 경우 그 개수)을 따지는 과정을 거쳐 이미지 속 모든 음식의 열량을 추정하는 모델을 설계하는 것은 더 도전적이다. 이미지가 나타내는 음식이 가지는 외관 상의 속성(열량에 초점을 맞추어)을 넘어 그 총량을 알 수 있다면 주어지는 이미지의 열량에 대해 더 높은 수준으로 대답할 수 있다.

# Our Method

## Dataset
Selenium 모듈을 활용하여 구글 이미지 검색을 통해 모델 학습에 효과적일 것으로 판단되는 10종류의 음식 이미지를 한·중·일식 별로 수집한다. 한식의 경우 한식 재단의 ‘한국인이 즐겨먹는 음식통계’를 참고하여 10종의 음식을 선정하였다. 중식과 일식의 경우 20종의 음식을 무작위로 선정한 후 음식 중 1회 제공량 당 칼로리를 계산하기 쉬우며 구글 검색 결과가 많은 10종의 음식을 선정하였다. <br><br>

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/68896078/09896ab3-3968-4126-b9d6-ad166ac36b3a)
<br><br>위와 같은 방법으로 선정된 음식은 다음과 같다.
<br>한식 10종: 칼국수, 짬뽕, 김밥, 비빔밥, 보쌈, 배추김치, 깍두기, 닭갈비, 김치볶음밥, 불고기
<br>일식 10종: nabe, soba, ramen, Japanese curry, sushi, udon, karaage, onigiri, gyudon, okonomiyaki
<br>중식 10종: congee, dong po rou, baozi, chaofan, zhajiangmian, sweet and sour pork, mapotofu, wonton soup, mooncake, pecking duck
<br>각 음식의 이미지를 약 400개씩 수집한 후 모델 학습에 적절하다고 판단되는 이미지 100여개를 직접 선정하였다.

## 이미지 데이터 수집
음식 종류를 결정한 후 구글 이미지 검색 결과를 저장하기 위해 ‘앱 애플리케이션 자동화를 위한 프레임워크’인 Selenium을 활용하였다.

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from urllib.request import (urlopen, urlparse, urlunparse, urlretrieve)
import urllib.request
import os
import pandas as pd
import json

chrome_path ='C:\Temp\chromedriver.exe' 
base_url = "http://www.google.co.kr/imghp?hl=ko"
```

더 많은 검색 결과를 저장하기 위해 scroll down 작업을 수행할 함수를 정의한다.
```python
def selenium_scroll_option():
  SCROLL_PAUSE_SEC = 3 # 스크롤을 내리는 동작 사이의 시간
  
  last_height = driver.execute_script("return document.body.scrollHeight")
  
  while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    time.sleep(SCROLL_PAUSE_SEC)
    
    new_height = driver.execute_script("return document.body.scrollHeight")
  
    if new_height == last_height:
        break
    last_height = new_height
```

음식의 이름을 key로 하고 검색어를 value로하는 dictinary를 선언한다.
```python
korean_foods = {"kalguksu":"칼국수", "champon":"짬뽕", "kimbap":"김밥", "bibimbap":"비빔밥", "bossam":"보쌈",
                "kimchi":"배추김치", "radish kimchi":"깍두기", "dak galbi":"닭갈비", "kimchi fried rice":"김치볶음밥", "bulgogi":"불고기"}
japanese_foods = {"nabe":"鍋", "soba":"そば", "ramen":"ラーメン", "japanese curry":"カレー","sushi":"寿司",
                  "udon":"うどん", "karaage":"唐揚げ","onigiri":"おにぎり","gyudon":"牛丼", "okonomiyaki":"お好み焼き"}
chinese_foods = {"congee":"粥", "dong po rou":"东坡肉", "baozi":"包子", "chaofan":"炒饭", "zhajiangmian":"炸酱面",
                  "sweet and sour pork":"糖醋肉", "mapotofu":"麻婆豆腐", "wonton soup":"馄饨汤", "mooncake":"月饼", "pecking duck":"烤鸭"}
```

해당 dictinary의 key를 이름으로 하는 디렉토리를 생성한 후 Dictionary의 value를 검색창에 입력한다. 더 많은 검색 결과를 위해 스크롤을 내린 후 이미지들의 url을 images_url 리스트에 저장한다.
```python
for i, j in korean_foods.items():
    image_name = i.replace(" ", "_")
    seach_word = j

    if not os.path.exists("./" + image_name):
        os.makedirs("./" + image_name)

    driver = webdriver.Chrome(chrome_path)
    driver.get('http://www.google.co.kr/imghp?hl=ko')
    browser = driver.find_element(By.NAME,"q") # 검색창 선택
    browser.send_keys(seach_word)
    browser.send_keys(Keys.RETURN)

    selenium_scroll_option()

    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd") 
    images_url = []
    for i in images: 
        if i.get_attribute('src')!= None :
                images_url.append(i.get_attribute('src'))
        else :
            images_url.append(i.get_attribute('data-src'))
```

중복되는 url을 제거하고 이미지를 다운로드한다.

```python
    print("전체 다운로드한 이미지 개수: {}\n동일한 이미지를 제거한 이미지 개수: {}".format(len(images_url), len(pd.DataFrame(images_url)[0].unique())))
    images_url=pd.DataFrame(images_url)[0].unique()
    
    for t, url in enumerate(images_url, 0):        
        urlretrieve(url, './' + image_name + '/' + image_name + '_' + str(t) + '.jpg')
    driver.close()
```

중·일식에 대하여 같은 작업을 수행한다.
```python
for i, j in chinese_foods.items():
  ...
```

## Naming & Labeling

다운받은 이미지 파일들의 이름을 변경한다.
```python

for i in korean_foods.keys():
    image_name = i.replace(" ", "_")
    
    file_path = "./" + image_name + '/'
    file_names = os.listdir(file_path)
    
    j = 0
    for name in file_names:
        src = file_path + name
        dst = os.path.join(file_path, i + '_' + str(j) + '.jpg')
        os.rename(src, dst)
        j+=1
```

Naming 작업이 완료되면 각각의 이미지에 해당 이미지의 id와 음식의 종류를 labeling 하는 작업을 수행한다.

```python
for i in korean_foods.keys():
    image_name = i.replace(" ", "_") 
    file_names = os.listdir('./'+image_name)

    tmp_json = [] 
   
    for j in file_names: 
        tmp_json.append({'image_id': j,'label': image_name}) 
    
    with open(i + '.json', 'w') as outfile: 
        json.dump(tmp_json, outfile, indent=4, sort_keys=True)
```

중·일식에 대하여 같은 작업을 수행한다.
```python
for i in chinese_foods.keys():
    ...
```



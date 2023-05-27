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

'''
from urllib.parse import quote_plus           
from bs4 import BeautifulSoup as bs  
import time
from urllib.request import (urlopen, urlparse, urlunparse, urlretrieve)
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import urllib.request
import os
import pandas as pd
'''

웹브라우저에서 이미지를 검색하도록 하기 위해 Selenium에서 webdriver를 import하고, 검색한 이미지의 URL을 일시적으로 dataframe 형태로 저장하기 위해 pandas를 import한다.

chrome_path ='C:\Temp\chromedriver.exe'
base_url = "https://www.google.co.kr/imghp"

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("lang=ko_KR")
chrome_options.add_argument('window-size=1920x1080')

driver = webdriver.Chrome(chrome_path,chrome_options=chrome_options)
driver.get(base_url)
driver.implicitly_wait(3)
driver.get_screenshot_as_file('google_screen.png')
driver.close()

Chrome webdriver를 driver로 이용하기 위해 불러오고 Google에서 이미지 검색을 위한 URL을 기본 위치로 설정한다.

def selenium_scroll_option():
  SCROLL_PAUSE_SEC = 3
  
  last_height = driver.execute_script("return document.body.scrollHeight")
  
  while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    time.sleep(SCROLL_PAUSE_SEC)
    
    new_height = driver.execute_script("return document.body.scrollHeight")
  
    if new_height == last_height:
        break
    last_height = new_height

검색 결과에 따라 나타나는 페이지에서 전체 이미지를 탐색하기 위해 scroll down을 수행할 함수를 정의한다.

japanese_foods = {"nabe":"鍋", "soba":"そば", "ramen":"ラーメン", "japanese_curry":"カレー","sushi":"寿司", "udon":"うどん", "karaage":"唐揚げ","onigiri":"おにぎり","gyudon":"牛丼", "okonomiyaki":"お好み焼き"}
chinese_foods = {"congee":"粥", "dong_po_rou":"东坡肉", "baozi":"包子", "chaofan":"炒饭", "zhajiangmian":"炸酱面", "sweet_and_sour_pork":"糖醋肉", "mapotofu":"麻婆豆腐", "wonton_soup":"馄饨汤", "mooncake":"月饼", "pecking_duck":"烤鸭"}

선별을 거쳐 검색하고자 하는 일식과 중식에 대한 검색어를 Dictionary 형식으로 저장한다.

for i, j in japanese_foods.items():
    image_name = i.replace(" ", "_")
    seach_word = j

    if not os.path.exists("./" + image_name):
        os.makedirs("./" + image_name)

    driver = webdriver.Chrome(chrome_path)
    driver.get('http://www.google.co.kr/imghp?hl=ko')
    browser = driver.find_element(By.NAME,"q")
    browser.send_keys(seach_word)
    browser.send_keys(Keys.RETURN)

    selenium_scroll_option()

'japanese_foods' Dictionary에 포함된 각각의 Key에 해당하는 Value를 검색어로 하여 japanese_foods의 모든 Key에 대해 Chromedriver를 이용한 이미지 검색을 실시한다.

    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd") 
    images_url = []
    for i in images: 
        if i.get_attribute('src')!= None :
                images_url.append(i.get_attribute('src'))
        else :
            images_url.append(i.get_attribute('data-src'))

검색 결과 얻은 이미지의 URL을 모두 저장한다.

    print("전체 다운로드한 이미지 개수: {}\n동일한 이미지를 제거한 이미지 개수: {}".format(len(images_url), len(pd.DataFrame(images_url)[0].unique())))
    images_url=pd.DataFrame(images_url)[0].unique()
    
    for t, url in enumerate(images_url, 0):        
        urlretrieve(url, './' + image_name + '/' + image_name + '_' + str(t) + '.jpg')
    driver.close()

중복되는 이미지의 URL을 제거하고 각 URL에서 이미지를 다운로드한다.

for i, j in chinese_foods.items():
  ...
  
chinese_foods에 포함된 Key에 대하여 같은 과정을 수행한다.

## Naming & Labelling

import os
import json

for i in japanese_foods.keys():
    image_name = i.replace(" ", "_")
    
    file_path = "./" + image_name + '/'
    file_names = os.listdir(file_path)

file_names를 생성한다.

    j = 0
    for name in file_names:
        src = file_path + name
        dst = os.path.join(file_path, i + '_' + str(j) + '.jpg')
        os.rename(src, dst)
        j+=1

이제부터 json 파일을 생성하기 위한 작업을 수행한다.

    file_names = os.listdir('./'+image_name)
    
'japanese_foods' Dictionary의 Key를 나열하여 생성한 'image_name' Directory 내의 파일들을 리스트 형식으로 file_names에 저장한다.

    tmp_json = []
   
    for j in file_names:
        tmp_json.append({'image_id': j, 'label': image_name})
    
json 파일을 생성하기 전 List 'tmp_json'에 file_names에 포함된 이미지의 이름과 Labelling한 해당 이미지의 종류를 저장한다.

    with open(i + '.json', 'w') as outfile:
        json.dump(tmp_json, outfile, indent=4, sort_keys=True)
        
tmp_json이 포함하는 정보를 json 형식으로 저장한다.

for i in chinese_foods.keys():
    ...

chinese_foods의 Key에 대하여 동일 작업을 수행한다.

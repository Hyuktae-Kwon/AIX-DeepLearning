# AI+X: Deep Learning

# 음식 이미지 인식을 통한 칼로리 계산

2019087792 권혁태 kwon0111@hanyang.ac.kr

2023090700 유시형 sihyeongyu9@gmail.com

2019082333 박영주 marchingwthu@gmail.com

**Contents**

I. Background & Scheme

II. Proposal

III. Our Method
<br>1. Dataset
<br>2. Methodology
<br>3. Result & Conclusion

IV. Related Work

# I. Background & Scheme
CNN 기법을 이용하여 서로 다른 이미지를 종류에 따라 분류하는 방법은 알려져 있다. 이를 기반으로 하여 모델이 인터넷에서 추출한 데이터셋을 바탕으로 학습한 결과를 근거로 처음 접하는 한식의 이미지를 보고 이미지에 해당하는 음식의 단위 제공량 당 열량을 추측하게끔 하고자 한다. Google image searching을 통해 얻은 중식과 일식의 검색량 상위에 해당하는 각 10종류의 음식 데이터셋에서 음식 종류별로 실제 단위 제공량 당 열량을 부여하고, 모델을 학습시킬 계획이다. 학습을 마친 모델이 주어진 한식의 이미지가 경험한 이미지 중 확률적으로 어떤 것에 가까운지 판단하여 해당 이미지가 indicating하는 음식의 단위 제공량 당 열량의 추정치를 결과로 내놓는 일련의 과정을 Tensorflow 내의 API, Keras를 이용한 CNN으로 구현하겠다.

# II. Proposal
본 프로젝트에서의 모델과 데이터셋을 이용해 처음 접하는 이미지가 이미 경험적으로 알고 있는 이미지 중 어떤 이미지와 유사한지 확률적인 근거를 가지고 판단했을 때 실제로 그 판단의 결과가 가지는 정확도에 대해 알아볼 수 있다. 동아시아 문화권에 속하여 대한민국과 일부 유사한 식문화를 가지는 일본과 중국의 음식 이미지를 학습시킨다. 학습 후 한식의 이미지를 접했을 때의 결과를 통해 새로운 이미지(여기서는 한식의)와 높은 관련도를 가지는 범주에 있는 음식에 대한 학습이 음식 열량의 추정에 유의미한 도움을 주는지 알고자 한다. 본 프로젝트에서는 음식(한식) 데이터에 대하여 이미지 속 음식의 양을 고려하지 않고 음식의 종류만을 추정하여 해당 음식의 열량을 결괏값으로 내놓는 모델을 설계할 것이다. 이미지에 담긴 음식의 양(개수를 셀 수 있는 음식의 경우 그 개수)을 따지는 과정을 거쳐 이미지 속 모든 음식의 열량을 추정하는 모델을 설계하는 것은 더 도전적이다. 이미지가 나타내는 음식이 가지는 외관 상의 속성(열량에 초점을 맞추어)을 넘어 그 총량을 알 수 있다면 주어지는 이미지의 열량에 대해 더 높은 수준으로 대답할 수 있다.

# III. Our Method

## 1. Dataset
Selenium 모듈을 활용하여 구글 이미지 검색을 통해 모델 학습에 효과적일 것으로 판단되는 10종류의 음식 이미지를 한·중·일식 별로 수집한다. 한식의 경우 한식 재단의 ‘한국인이 즐겨먹는 음식통계’를 참고하여 10종의 음식을 선정하였다. 중식과 일식의 경우 20종의 음식을 무작위로 선정한 후 음식 중 1회 제공량 당 칼로리를 계산하기 쉬우며 구글 검색 결과가 많은 10종의 음식을 선정하였다. <br><br>

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/68896078/09896ab3-3968-4126-b9d6-ad166ac36b3a)
<br><br>위와 같은 방법으로 선정된 음식은 다음과 같다.
<br>한식 10종: 칼국수, 짬뽕, 김밥, 비빔밥, 보쌈, 배추김치, 깍두기, 닭갈비, 김치볶음밥, 불고기
<br>일식 10종: nabe, soba, ramen, Japanese curry, sushi, udon, karaage, onigiri, gyudon, okonomiyaki
<br>중식 10종: congee, dong po rou, baozi, chaofan, zhajiangmian, sweet and sour pork, mapotofu, wonton soup, mooncake, pecking duck
<br>각 음식의 이미지를 약 400개씩 수집한 후 모델 학습에 적절하다고 판단되는 이미지 100여개를 직접 선정하였다.

### 이미지 데이터 수집
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

음식의 이름을 key로 하고 검색어를 value로하는 dictionary를 선언한다.
```python
korean_foods = {"kalguksu":"칼국수", "champon":"짬뽕", "kimbap":"김밥", "bibimbap":"비빔밥", "bossam":"보쌈",
                "kimchi":"배추김치", "radish kimchi":"깍두기", "dak galbi":"닭갈비", "kimchi fried rice":"김치볶음밥", "bulgogi":"불고기"}
japanese_foods = {"nabe":"鍋", "soba":"そば", "ramen":"ラーメン", "japanese curry":"カレー","sushi":"寿司",
                  "udon":"うどん", "karaage":"唐揚げ","onigiri":"おにぎり","gyudon":"牛丼", "okonomiyaki":"お好み焼き"}
chinese_foods = {"congee":"粥", "dong po rou":"东坡肉", "baozi":"包子", "chaofan":"炒饭", "zhajiangmian":"炸酱面",
                  "sweet and sour pork":"糖醋肉", "mapotofu":"麻婆豆腐", "wonton soup":"馄饨汤", "mooncake":"月饼", "pecking duck":"烤鸭"}
```

해당 dictionary의 key를 이름으로 하는 디렉토리를 생성한 후 dictionary의 value를 검색창에 입력한다. 더 많은 검색 결과를 위해 스크롤을 내린 후 이미지들의 url을 images_url 리스트에 저장한다.
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
    print("전체 다운로드한 이미지 개수: {}\n동일한 이미지를 제거한 이미지 개수: {}"
         .format(len(images_url), len(pd.DataFrame(images_url)[0].unique())))
    images_url=pd.DataFrame(images_url)[0].unique()
    
    for t, url in enumerate(images_url, 0):        
        urlretrieve(url, './' + image_name + '/' + image_name + '_' + str(t) + '.jpg')
    driver.close()
```

중·일식에 대하여 같은 작업을 수행한다.
```python
for i, j in chinese_foods.items():
    ...
  
for i, j in japanese_foods.items():
    ...
```

### Naming & Labeling
다운로드 이미지 파일들의 이름을 변경한다.
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

for i in japanese_foods.keys():
    ...
```

## 2. Methodology

### 이미지 전처리
```python
import os
from PIL import Image
from numpy import expand_dims
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

korean_foods = {"kalguksu":"칼국수", "champon":"짬뽕", "kimbap":"김밥", "bibimbap":"비빔밥", "bossam":"보쌈",
                "kimchi":"배추김치", "radish kimchi":"깍두기", "dak galbi":"닭갈비", "kimchi fried rice":"김치볶음밥", "bulgogi":"불고기"}
japanese_foods = {"nabe":"鍋", "soba":"そば", "ramen":"ラーメン", "japanese curry":"カレー","sushi":"寿司",
                  "udon":"うどん", "karaage":"唐揚げ","onigiri":"おにぎり","gyudon":"牛丼", "okonomiyaki":"お好み焼き"}
chinese_foods = {"congee":"粥", "dong po rou":"东坡肉", "baozi":"包子", "chaofan":"炒饭", "zhajiangmian":"炸酱面",
                  "sweet and sour pork":"糖醋肉", "mapotofu":"麻婆豆腐", "wonton soup":"馄饨汤", "mooncake":"月饼", "pecking duck":"烤鸭"}
```

이미지의 크기를 통일하고, 'image_name_resized' directory에 크기 조정된 이미지를 저장한다.
```python
for i, j in korean_foods.items():
    image_name = i.replace(" ", "_")
    file_path = "./" + image_name + '/'
    file_names = os.listdir(file_path)

    for f in file_names:
        img = Image.open(file_path + f)
        resized_img = img.resize((256, 256))

        if not os.path.exists("./" + image_name + "_resized"): 
            os.makedirs("./" + image_name + "_resized")

	title, ext = os.path.splitext(f)
        resized_img.save("./" + image_name + "_resized/" + title + "_r" + ext)

```

데이터 전처리를 실시한다.
```python
for i, j in korean_foods.items():
    image_name = i.replace(" ", "_")
    file_path = "./" + image_name + "_resized" + '/'
    file_names = os.listdir(file_path)
    print(file_names)
    for f in file_names:
        img = load_img(file_path + f)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(
                                    zoom_range=[0.8, 1.0],
                                    rotation_range=45,
                                    brightness_range=[0.3, 1.2],
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    height_shift_range=0.3,
                                    width_shift_range=0.3
        )

# ImageDataGenerator로 변경된 이미지 확인
    it = datagen.flow(samples, batch_size=1)
    fig = plt.figure(figsize=(20,20))
    plt.title(i)
    for i in range(12):
        plt.subplot(4, 3, 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)
    plt.show()
```

### Data ugmentation
Data augmentation을 준비한다.
```python
from glob import glob
from PIL import Image
image_datas = glob('./korean_foods/*/*.jpg')
for imagename in image_datas:
    image = Image.open(imagename)
    lr_image = image.transpose(Image.FLIP_LEFT_RIGHT) # 좌우 반전
    lr_imagename = imagename.replace(".jpg", "") + " " + ".jpg"
    ud_image = image.transpose(Image.FLIP_TOP_BOTTOM) # 상하 반전
    ud_imagename = imagename.replace(".jpg", "") + "  " + ".jpg"

    lr_image.save(lr_imagename)
    ud_image.save(ud_imagename)
```

수집한 이미지 각각에 대응하는 좌우 반전 이미지와 상하 반전 이미지를 생성하여 데이터셋의 크기를 세 배로 키운다.
```python
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from glob import glob\n",
    "from PIL import Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "image_datas = glob('./korean_foods/*/*.jpg')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "for imagename in image_datas:\n",
    "    image = Image.open(imagename)\n",
    "    lr_image = image.transpose(Image.FLIP_LEFT_RIGHT)       #좌우 반전\n",
    "    lr_imagename = imagename.replace(\".jpg\", \"\") + \" \" + \".jpg\"\n",
    "    ud_image = image.transpose(Image.FLIP_TOP_BOTTOM)       #상하 반전\n",
    "    ud_imagename = imagename.replace(\".jpg\", \"\") + \"  \" + \".jpg\"\n",
    "\n",
    "    lr_image.save(lr_imagename)\n",
    "    ud_image.save(ud_imagename)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "foodkcal",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.16"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

중, 일식의 데이터셋에 대하여 Data augmentation을 수행한다.
```python
...
"image_datas = glob('./chinese_foods/*/*.jpg')"
...

...
"image_datas = glob('./japanese_foods/*/*.jpg')"
...
```

### Modeling & Training
```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import re
import os
```

```python
image_datas = glob('./korean_foods/*/*.jpg')
class_name = ["kalguksu", "champon", "kimbap", "bibimbap", "bossam",
                "kimchi", "radish_kimchi", "dak_galbi", "kimchi_fried_rice", "bulgogi"]
dic = {"kalguksu":0, "champon":1, "kimbap":2, "bibimbap":3, "bossam":4,
                "kimchi":5, "radish_kimchi":6, "dak_galbi":7, "kimchi_fried_rice":8, "bulgogi":9}
```

```python
X = []
Y = []
for imagename in image_datas:
    image = Image.open(imagename)
    image = image.resize((128, 128))
    image = np.array(image)
    X.append(image)
    label = imagename.split('\\')[2].replace('.jpg','').replace(' ','')
    label = re.sub(r"[0-9]", "", label)
    label = label[:-1]
    label = dic[label]
    Y.append(label)
```

```python
X = np.array(X)    
Y = np.array(Y)
```

```python
train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, 
                                                    shuffle=True, random_state=44)

train_labels = train_labels[..., tf.newaxis]
test_labels = test_labels[..., tf.newaxis]

train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
```

training set의 각 class 별 image 수를 확인한다.
```python
unique, counts = np.unique(np.reshape(train_labels, (866,)), axis=-1, return_counts=True)
dict(zip(unique, counts))
```

test set의 각 class 별 image 수를 확인한다.
```python
unique, counts = np.unique(np.reshape(test_labels, (217,)), axis=-1, return_counts=True)
dict(zip(unique, counts))
```

```python
N_TRAIN = train_images.shape[0]
N_TEST = test_images.shape[0]
```

데이터를 확인한다.
```python
plt.figure(figsize=(15,9))
for i in range(15):
    img_idx = np.random.randint(0, 875)
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[img_idx])
    plt.xlabel(class_name[train_labels[img_idx][0]])
```

pixel 값을 0~1 사이로 조정한다.
```python
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.
```

label을 onehot-encoding 한다.
```python
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
```

```python
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)
```

```python
learning_rate = 0.0001
N_EPOCHS = 50
N_BATCH = 40
N_CLASS = 10
```

Dataset을 구성한다.
```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
                buffer_size=875).batch(N_BATCH).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
				N_BATCH)
```

Sequential API를 사용하여 Model을 구성한다.
```python
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, 
                                  activation='relu', padding='SAME', 
                                  input_shape=(128, 128, 3)))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, 
                                  activation='relu', padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, 
                                  activation='relu', padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
```

Model을 생성하고 compiling한다.
```python
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

```python
steps_per_epoch = N_TRAIN//N_BATCH
validation_steps = N_TEST//N_BATCH
print(steps_per_epoch, validation_steps)
```

training 실시한다.
```python
history = model.fit(train_dataset, epochs=N_EPOCHS, steps_per_epoch=steps_per_epoch, 
                    validation_data=test_dataset, validation_steps=validation_steps)
```

```python
model.evaluate(test_dataset)
```

loss에 대한 plotting을 실시한다.
```python
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```
![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/e539bc0b-3d05-4357-b5b0-47263eabf037)

accuracy에 대한 plotting을 실시한다.
```python
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```
![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/0cf78d09-bb73-4281-9a73-13d182c44e1e)

같은 방식으로 중식과 일식의 이미지를 학습하고 트레이닝 횟수에 따라 loss와 accuracy를 plotting한 결과는 다음과 같다.

**중식 이미지 학습 결과**

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/12c75843-6b59-4785-a0ae-652497a21f04)

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/e8e8d395-6994-4b19-9dfb-7c44abb2ed72)

**일식 이미지 학습 결과**

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/311fb7c7-1f67-44bb-9381-b1ca03c0adc3)

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/af2ab9dd-31d9-4a12-8e69-2cb35201cfe4)

### Test
한식 샘플 이미지가 나타내는 음식의 종류를 추정한다. 모델은 학습의 결과를 바탕으로 '각 이미지가 나타내는 음식'이 '학습한 음식'과 갖는 유사도에 따라 '학습한 음식'에 대하여 갖는 기댓값을 산출한다. 기댓값을 산출할 때 '학습한 음식'으로 이루어지는 후보군 내의 모든 후보 음식에 대한 기댓값을 모두 더하면 1(100%)이 된다. 기댓값을 막대그래프로 나타내었다.
```python
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_name[predicted_label],
                                100*np.max(predictions_array),
                                class_name[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(N_CLASS), class_name, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(N_CLASS), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
```

```python
rnd_idx = np.random.randint(1, N_TEST//N_BATCH)
img_cnt = 0
for images, labels in test_dataset:
    img_cnt += 1
    if img_cnt != rnd_idx:
        continue
    predictions = model(images, training=False)
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    labels = tf.argmax(labels, axis=-1)
    plt.figure(figsize=(3*2*num_cols, 4*num_rows))
    plt.subplots_adjust(hspace=1.0)
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions.numpy(), labels.numpy(), images.numpy())
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions.numpy(), labels.numpy())        
    break
```
![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/381a0aa7-6de9-4a5c-b52f-3486bd043be6)

중식 샘플 이미지가 나타내는 음식의 종류를 추정한다.

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/24e66c6b-8ddb-4a27-ad2b-5d8ab3381f35)

일식 샘플 이미지가 나타내는 음식의 종류를 추정한다.

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/aa57f844-542a-4df5-a36a-2cf716959917)

## 3. Result & Conclusion
열량 추정을 위하여 임의로 6종류의 한식(Bindaetteok, cold_noodles, japchae, pork_barbecue, tteokbokki, yukgaejang)을 선정하였다. 각 '학습한 음식'의 열량을 Q, 선정한 6종류의 음식을 나타내는 어떤 이미지가 각 '학습한 음식'에 대하여 갖는 기댓값을 E라고 할 때 그 이미지가 나타내는 음식에 대하여 모델이 추정하는 열량은 다음과 같다.

$$\sum_{i=1}^N Q_iE_i$$

임의로 선정한 6종류의 음식이 갖는 실제 열량은 다음과 같다.

|음식|열량|
|---|---|
|Bindaetteok|194kcal|
|cold_noodles|450kcal|
|japchae|191kcal|
|pork_barbecue|415kcal|
|tteokbokki|304kcal|
|yukgaejang|165kcal|

앞서 10종류의 한식으로 학습시킨 결과를 바탕으로 하여, 제시한 6종류의 한식의 열량을 추정한다.
```python
cur_dir = os.getcwd()
ckpt_dir = 'checkpoints'
file_name = 'korean_cnn_weights.h5'

dir = os.path.join(cur_dir, ckpt_dir)
os.makedirs(dir, exist_ok=True)

file_path = os.path.join(dir, file_name)
```

```python
model.save(file_path)
```

```python
food_kcal = {"kalguksu":420, "champon":688, "kimbap":485, "bibimbap":586, "bossam":1296,
                "kimchi":29, "radish_kimchi":19, "dak_galbi":669, "kimchi_fried_rice":530, "bulgogi":471}
```

```python
testimg_datas = glob('./test_img/*.jpg')

X2 = []
Y2 = []
for testimgname in testimg_datas:
    image = Image.open(testimgname)
    image = image.resize((128, 128))
    image = np.array(image)
    X2.append(image)
    label = testimgname.split('\\')[1].replace('.jpg','').replace(' ','')
    Y2.append(label)

X2 = np.array(X2)  

X2 = X2.astype(np.float32) / 255.

predictions = model(X2, training=False)

num_rows = 2
num_cols = 3

def calculate_kcal(predict_array):
    kcal = 0
    for i in range(10):
        kcal += predict_array[i] * food_kcal[class_name[i]]
    return kcal

plt.figure(figsize=(3*2*num_cols, 4*num_rows))
plt.subplots_adjust(hspace=1.0)

for i in range(len(X2)):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X2[i])
    plt.xlabel("{} \nexpected : {:2.0f}kcal".format(Y2[i], calculate_kcal(predictions.numpy()[i])))
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plt.grid(False)    
    plt.xticks(range(N_CLASS), class_name, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(N_CLASS), predictions.numpy()[i], color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions.numpy()[i])
    thisplot[predicted_label].set_color('red')

    print(Y2[i], calculate_kcal(predictions.numpy()[i]))
```
![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/b438ad63-bb4d-47cc-a6d4-37fe3968bb24)

중식으로 학습을 거친 후 제시한 6종류의 한식의 열량을 추정한 결과는 다음과 같다.

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/a1f2cb64-bc88-46f2-a895-34ac1f2ae35f)

일식으로 학습을 거친 후 제시한 6종류의 한식의 열량을 추정한 결과는 다음과 같다.

![image](https://github.com/kwon-0111/AIX-DeepLearning/assets/132051184/83fa3fff-2386-45be-8faa-954ee5daee1a)

한식, 중식, 일식 각각의 데이터셋으로 학습을 거친 후 제시한 6종류의 한식에 대하여 추정한 열량의 백분율 상대오차를 표로 나타내었다.

**한식 데이터셋으로 학습을 거친 후 열량을 추정한 결과 및 백분율 상대오차**

|음식|결과(오차)|
|---|---|
|Bindaetteok|1073.38kcal(+453.29%)|
|cold_noodles|505.33kcal(+12.30%)|
|japchae|642.05kcal(+236.15)|
|pork_barbecue|209.69kcal(-49.47%)|
|tteokbokki|110.32kcal(-63.71%)|
|yukgaejang|606.25kcal(+267.42%)|

**중식 데이터셋으로 학습을 거친 후 열량을 추정한 결과 및 백분율 상대오차**

|음식|결과(오차)|
|---|---|
|Bindaetteok|470.34kcal(+142.44%)|
|cold_noodles|245.08kcal(-45.54%)|
|japchae|820.54kcal(+329.60%)|
|pork_barbecue|649.21kcal(+56.44%)|
|tteokbokki|481.19kcal(+58.28%)|
|yukgaejang|444.62kcal(+169.47%)|

**일식 데이터셋으로 학습을 거친 후 열량을 추정한 결과 및 백분율 상대오차**

|음식|결과(오차)|
|---|---|
|Bindaetteok|498.82kcal(+157.12%)|
|cold_noodles|569.11kcal(+26.47%)|
|japchae|480.60kcal(+151.62%)|
|pork_barbecue|412.35kcal(-0.64%)|
|tteokbokki|426.51kcal(+40.30%)|
|yukgaejang|518.80kcal(+214.42%)|

한식 데이터셋으로 학습한 후에 새로운 이미지가 나타내는 음식(한식)의 열량을 추정했을 때, 중식과 일식 데이터셋으로 학습한 후 열량을 추정했을 때보다 높은 정확도를 보여주지 않았다. 한식이 가지는 중식 및 일식과의 차이점이 학습 과정에서 모델에게 인식되지 않았거나 각 데이터셋 종류 별 학습 후 추정 결과에서 유의미한 정확도 차이가 나타날 만큼 한식, 중식, 일식이 구분되지 않는다.

**Work Distributed**

권혁태: dataset processing, write up
<br>유시형: code implementation
<br>박영주: write up

# IV. Related Work

**Libraries and Modules Used for the Work**

selenium
<br>urllib.request
<br>os
<br>pandas
<br>json
<br>PIL
<br>numpy
<br>tensorflow
<br>matplotlib
<br>glob
<br>sklearn.model_selection
<br>re

**Blogs Used for the Work**

https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=isu112600&logNo=221582003889

https://mj-lahong.tistory.com/82

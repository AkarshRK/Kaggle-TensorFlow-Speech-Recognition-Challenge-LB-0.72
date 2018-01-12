import os,errno
import cv2
import numpy as np
import random
from collections import Counter
path = r"\Python36\train\audio"
D = {'down':0, 'go':1,'left':2,'no':3,'off':4,'on':5,'right':6,'stop':7,'up':8,'yes':9,'silence':10}

def split_list(a_list):
    half = int(len(a_list)/2)
    return a_list[:half], a_list[half:]

def vectorized_result(j):
  e = np.zeros((1,12))
  if(str(j) in D):
    e[0][D[j] ] = 1.0
  else:
    e[0][11] = 1.0
  return e

categories = os.listdir(path)
categories.remove('_background_noise_')

speakers = []
for speech in categories:
 navigate = path + "\\" + speech
 audiofiles = os.listdir(navigate)
 audiofiles = [x[0:len(x)-5] for x in audiofiles]
 for audio in audiofiles:
   speakers.append(audio)

speakers = Counter(speakers)
speakers = dict(speakers)

sg30 = []
sl30 = []
for k,v in speakers.items():
 if(v>30):
  sg30.append(k)
 else:
  sl30.append(k)

trainX = []
trainY = []
validateX = []
validateY = []
testX = []
testY = [] 


path_original = r"\Python36\Augmented_Data\Original_data"
path_leftshift = r"\Python36\Augmented_Data\LeftShift_data"
path_rightshift = r"\Python36\Augmented_Data\RightShift_data"

path_original_bg = r"\Python36\Augmented_Data\Data_with_bg\Original_bg"
path_leftshift_bg = r"\Python36\Augmented_Data\Data_with_bg\LeftShift_bg"
path_rightshift_bg = r"\Python36\Augmented_Data\Data_with_bg\RightShift_bg"

path = {"path_original":r"\Python36\Augmented_Data\Original_data"}

for k,v in path.items():
 print("--",k,"--")
 for speech in categories:
   navigate = v + "\\" + speech
   pictures = os.listdir(navigate)
   one_hot_vector = vectorized_result(speech)
   for pic in pictures:
     bgr = cv2.imread((navigate + "\\" + pic),cv2.IMREAD_COLOR)
     norm_image = bgr
     norm_image = cv2.normalize(bgr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,dst=norm_image)
     #norm_image = np.delete(norm_image,0,axis=2)
     if pic[0:len(pic)-9] in sl30:
      trainX.append(norm_image)
      trainY.append(one_hot_vector)
     elif pic[0:len(pic)-9] in sg30:
      validateX.append(norm_image)
      validateY.append(one_hot_vector)
   print(speech +" is done ")

path = {"path_leftshift":r"\Python36\Augmented_Data\LeftShift_data",
        "path_rightshift":r"\Python36\Augmented_Data\RightShift_data",
        "path_original_bg":r"\Python36\Augmented_Data\Data_with_bg\Original_bg"}

for k,v in path.items():
 print("--",k,"--")
 for speech in categories:
   navigate = v + "\\" + speech
   pictures = os.listdir(navigate)
   random.shuffle(pictures)
   pictures = pictures[0:600]
   one_hot_vector = vectorized_result(speech)
   for pic in pictures:
     bgr = cv2.imread((navigate + "\\" + pic),cv2.IMREAD_COLOR)
     norm_image = bgr
     norm_image = cv2.normalize(bgr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,dst=norm_image)
     #norm_image = np.delete(norm_image,0,axis=2)
     if pic[0:len(pic)-9] in sl30:
      trainX.append(norm_image)
      trainY.append(one_hot_vector)
     elif pic[0:len(pic)-9] in sg30:
      validateX.append(norm_image)
      validateY.append(one_hot_vector)
   print(speech +" is done ")

   


path = r"\Python36\Augmented_Silence_Data"
pictures  =os.listdir(path + "\\" + "Train")
random.shuffle(pictures)
count = 0
one_hot_vector = vectorized_result('silence')
for pic in pictures:
 bgr = cv2.imread((path + "\\" + "Train" + "\\" + pic),cv2.IMREAD_COLOR)
 norm_image = bgr
 norm_image = cv2.normalize(bgr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,dst=norm_image)
 #norm_image = np.delete(norm_image,0,axis=2)
 if count <5000:
  trainX.append(norm_image)
  trainY.append(one_hot_vector)
  count = count + 1
 else:
  validateX.append(norm_image)
  validateY.append(one_hot_vector)


from numpy import array
trainX = array(trainX)
trainY = array(trainY)
testX = array(testX)
testY = array(testY)
validateX = array(validateX)
validateY = array(validateY)

trainY = trainY.reshape(-1,12)
testY = testY.reshape(-1,12)
validateY =validateY.reshape(-1,12)

perm = np.random.permutation(len(trainX))
trainX = np.take(trainX,perm,axis=0)
trainY = np.take(trainY,perm,axis=0)

perm = np.random.permutation(len(testX))
testX = np.take(testX,perm,axis=0)
testY = np.take(testY,perm,axis=0)

perm = np.random.permutation(len(validateX))
validateX = np.take(validateX,perm,axis=0)
validateY = np.take(validateY,perm,axis=0)












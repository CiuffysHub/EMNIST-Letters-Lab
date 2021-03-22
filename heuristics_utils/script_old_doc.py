print('importing...')
import os
import gzip
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data,exposure,measure
from sklearn.metrics import plot_confusion_matrix
from sklearn import decomposition
from pyefd import elliptic_fourier_descriptors  # this one is not in anaconda navigator

def load_data(file):

    data = np.load(file)
    data = data['arr_0']

    return data

tmp=65
real_labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','|']
dim_train = 124800 # 124800
dim_test = 20800 # 20800


def calculate_hog(features,save=True,orientations=8, pixels_per_cell=(8,8),cells_per_block=(2,2),cut=True):
  
  global tmp
  features = np.reshape(features,(len(features),28,28))
  features_copy = features.copy()
  features = np.zeros(shape=(len(features),128))

  for i in range(len(features)):
    features[i], image = hog(features_copy[i], orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block, visualize=True, multichannel=False)
  if cut is True:
    for i in range(len(features)):
      features[i] = (features[i]*200)//2

  if save is True:
    tmp = tmp+1
    np.savez_compressed('tmp'+chr(tmp), features)

  return features

def confusion_matrix(classifier, features, labels):
  disp = plot_confusion_matrix(classifier, features, labels,
                             display_labels=real_labels,
                             cmap=plt.cm.Blues)
  plt.show()

def pca(features, n_components):
  pca = decomposition.PCA(n_components=n_components)
  pca.fit(features)
  features = pca.transform(features)
  return features

def load_fourier(order=3):
  train = load_data('fourier_train.npz')[:dim_train]
  test = load_data('fourier_test.npz')[:dim_test]

  train = [(x*300//3)[:order*4-3] for x in train]
  test = [(x*300//3)[:order*4-3] for x in test]

  return train, test

def random_forest(train, test, labels_train, labels_test, est=10):
  print('training RF...')
  rfc = RandomForestClassifier(n_jobs=-1, n_estimators=est)
  rfc.fit(train, labels_train)
  print('scoring RF...')
  print(rfc.score(test, labels_test))
  return rfc


def decision_tree(train, test, labels_train, labels_test):
  print('training DT...')
  dtree = DecisionTreeClassifier()
  dtree.fit(train, labels_train)
  print('scoring DT...')
  print(dtree.score(test, labels_test))
  return dtree


print('Indexing...')

pca_train, pca_test = load_fourier(4)

pca_train2 = load_data('projection_train.npz')[:dim_train]
pca_test2 = load_data('projection_test.npz')[:dim_test]

pca_train2=pca(pca_train2,7)
pca_test2=pca(pca_test2,7)

print('combining...')

features_train = [list(pca_train[i])+list(pca_train2[i]) for i in range(0,dim_train)]
features_test = [list(pca_test[i])+list(pca_test2[i]) for i in range(0,dim_test)]



labels_train = load_data('labels_train_il.npz')[:dim_train]
labels_test = load_data('labels_test_il.npz')[:dim_test]

np.savez_compressed('labels_il', test=labels_test,train=labels_train)

print('training start!')

c = random_forest(features_train, features_test, labels_train, labels_test, 150)
confusion_matrix(c, features_test, labels_test)

'''
Question: is n_estimators = 1 just a Decision Tree?
Accuracy looks close but RF=1 is much faster, maybe it's missing
some kind of "somewhat not very effective" optimization that takes much longer

Heuristic  size       acc     pca
original:  784        83%     45%,7
profiles:  112        82%     58%,12
fourier:   9(order=3) 72%     0%
hog (pdf): 128        80%     46%,15
crossings: 56         67%     61%,9
projection:56         72%     60%,8
ori+fou:   16                 78%
prj+fou:   17                 81%
cro+fou:   18                 81%
pf+pr+f:   29                 81%
cro+prf:   21                 75%
prf+fou:   21                 80%
hog+fou:   24                 70%

neighborhood analysis si blocca
idealmente vogliamo 3 features di cui fare il grafico tridimensionale
pca is affected by preprocessing positively

300 is good for the 10.000 numbers in mnist
let's try (prj+fou):
10: 81%
150: 85%
300: 85%
600: 85%

150 with real features is pretty slow (heuristics matter!) and gives us 88%, we get to 90% with the addition of the | feature (at 150)
and 86% if reduced in size with best heuristics

'''


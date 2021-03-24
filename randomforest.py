print('Importing...')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import time
import sys

max = 0
real_labels = []

def random_forest(train, test, labels_train, labels_test, est=150):
  print('Training RF...')
  rfc = RandomForestClassifier(n_jobs=-1, n_estimators=est)
  rfc.fit(train, labels_train)
  print('Scoring RF...')
  print(rfc.score(test, labels_test))
  return rfc

if __name__ == "__main__":
  features = sys.argv[1]
  labels = sys.argv[2]
  
  print('Loading features...')

  data = np.load(features)
  train_set = data['train']
  test_set = data['test']

  print('Loading labels...')

  data = np.load(labels)
  train_labels = data['train']
  test_labels = data['test']
  for l in test_labels:
  	if l>max:
  		max=l;

  print('Starting training...')

  start = time.perf_counter()
  trees = int(sys.argv[3])
  classifier = random_forest(train_set, test_set, train_labels, test_labels, trees)
  print('Done in',time.perf_counter()-start,'seconds!')
  real_labels = [chr(x) for x in range(65,91)]
  if max == 27:
  	real_labels=real_labels+['|']

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,9))

  disp = plot_confusion_matrix(classifier, test_set, test_labels,
                             display_labels=real_labels,
                             cmap=plt.cm.Blues,ax=axes.flatten()[0])

  right = [0]*max
  wrong = [0]*max
  predicts = classifier.predict(test_set)
  np.savez_compressed('predicts',preds=classifier.predict(test_set))
  for i in range(0,len(test_set)):
    if test_labels[i] == predicts[i]:
      right[test_labels[i]-1]=right[test_labels[i]-1]+1
    else:
      wrong[test_labels[i]-1]=wrong[test_labels[i]-1]+1


  X = np.arange(max)

  plt.bar(X, np.asarray(right), color = 'r')
  plt.bar(X, -np.asarray(wrong), color = 'b')
  plt.xticks(X, real_labels)
  plt.show()
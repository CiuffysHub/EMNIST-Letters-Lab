print('Importing...')

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import time
import sys

max = 0
real_labels = []

def decision_tree(train, test, labels_train, labels_test):
  print('Training DT...')
  dtree = DecisionTreeClassifier()
  dtree.fit(train, labels_train)
  print('Scoring DT...')
  print(dtree.score(test, labels_test))
  return dtree

def random_forest(train, test, labels_train, labels_test, est=150):
  print('Training RF...')
  rfc = RandomForestClassifier(n_jobs=-1, n_estimators=est)
  rfc.fit(train, labels_train)
  print('Scoring RF...')
  print(rfc.score(test, labels_test))
  return rfc

def confusion_matrix(classifier, features, labels):
  disp = plot_confusion_matrix(classifier, features, labels,
                             display_labels=real_labels,
                             cmap=plt.cm.Blues)
  plt.show()


if __name__ == "__main__":
  features = sys.argv[1]
  labels = sys.argv[2]
  tree_type = sys.argv[3]
  
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
  if (tree_type == "random_forest"):
  	trees = int(sys.argv[4])
  	classifier = random_forest(train_set, test_set, train_labels, test_labels, trees)
  if (tree_type == "decision_tree"):
  	classifier = decision_tree(train_set, test_set, train_labels, test_labels)
  print('Done in',time.perf_counter()-start,'seconds!')
  real_labels = [chr(x) for x in range(65,91)]
  if max == 27:
  	real_labels=real_labels+['|']
  confusion_matrix(classifier, test_set, test_labels)
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
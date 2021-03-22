import numpy as np
from sklearn import decomposition
import sys

def pca(features, n_components):
  pca = decomposition.PCA(n_components=n_components)
  pca.fit(features)
  features = pca.transform(features)
  return features

if __name__ == "__main__":
  features = np.load(sys.argv[1])
  features_train = features['train']
  features_test = features['test']
  pca_n = int(sys.argv[2])
  pca_train = pca(features_train,pca_n)
  pca_test = pca(features_test,pca_n)
  np.savez_compressed(sys.argv[3],train=pca_train,test=pca_test)
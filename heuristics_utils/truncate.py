import numpy as np
import sys

def truncate(features, n_components):
  features = (features*2*(10**n_components))//2
  return features

if __name__ == "__main__":
  features = np.load(sys.argv[1])
  features_train = features['train']
  features_test = features['test']
  truncate_n = int(sys.argv[2])
  truncate_train = truncate(features_train,truncate_n)
  truncate_test = truncate(features_test,truncate_n)
  np.savez_compressed(sys.argv[3],train=truncate_train,test=truncate_test)
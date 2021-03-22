import numpy as np
import sys

def calculate_profiles(features):
  print('reshaping...')
  features = features.copy()
  features = np.reshape(features,(len(features),28,28))
  features_profiles = np.zeros((len(features),112))
  features = np.asarray(features)

  print('processing...')
  for im in range(0,len(features)):
    print('progress:',im,'/',len(features))
    for i in range(0,4):
      for row in range(0,28):
        c = 0
        for pixel in range(1,28):
          if features[im][row][pixel]!=0 and features[im][row][pixel-1]==0:
            break
          c = c+1
        features_profiles[im][row+28*i] = c
      features[im] = np.asarray(list(zip(*features[im][::-1])))

  return features_profiles

if __name__ == "__main__":
  features = np.load(sys.argv[1])
  features_train = features['train']
  features_test = features['test']
  prof_train = calculate_profiles(features_train)
  prof_test = calculate_profiles(features_test)
  np.savez_compressed(sys.argv[2],train=prof_train,test=prof_test)

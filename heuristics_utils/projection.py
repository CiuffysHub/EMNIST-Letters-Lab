import numpy as np
import sys

def calculate_projection(features):
  print('reshaping...')
  features = features.copy()
  features = np.reshape(features,(len(features),28,28))
  features_projection = np.zeros((len(features),56))

  print('processing...')
  for im in range(0,len(features)):
    print('progress:',im,'/',len(features))
    for i in range(0,2):
      for row in range(0,28):
        c = 0
        for pixel in range(0,28):
          if features[im][row][pixel]!=0:
            c = c+1
        features_projection[im][row+28*i] = c
      features[im] = features[im].transpose()

  return features_projection

if __name__ == "__main__":
  features = np.load(sys.argv[1])
  features_train = features['train']
  features_test = features['test']
  projection_train_features = calculate_projection(features_train)
  projection_test_features = calculate_projection(features_test)
  np.savez_compressed(sys.argv[2],train=projection_train_features,test=projection_test_features)
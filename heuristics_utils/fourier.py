from pyefd import elliptic_fourier_descriptors
from skimage import measure
import numpy as np
import sys

def calculate_fourier(features,order=15):
  features = np.reshape(features,(len(features),28,28))
  features_fourier = np.zeros(shape=(len(features),order*4-3))
  for i in range(len(features)):
    contours = measure.find_contours(features[i],level=0.8)
    coeffs = elliptic_fourier_descriptors(contours[0], order=order, normalize=True)
    feature = ((coeffs.flatten()[3:])*300)//3
    features_fourier[i]=feature
    print('progress:',i,'/',len(features))
  return features_fourier

if __name__ == "__main__":
  features = np.load(sys.argv[1])
  features_train = features['train']
  features_test = features['test']
  fourier_train_features = calculate_fourier(features_train,order=4)
  fourier_test_features = calculate_fourier(features_test,order=4)
  np.savez_compressed(sys.argv[2],train=fourier_train_features,test=fourier_test_features)
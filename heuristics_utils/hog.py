from skimage.feature import hog
import numpy as np
import sys


def calculate_hog(features):
	features_hog = []
	c = 0
	for feature in features:
		c=c+1
		image = feature.reshape((28,28))

		fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
		                cells_per_block=(2, 2), visualize=True, multichannel=False)
		features_hog.append((fd*200)//2)
		print('progress:',c,'/',len(features))
	return features_hog

if __name__ == "__main__":
  features = np.load(sys.argv[1])
  path = sys.argv[2]
  features_train = features['train']
  features_test = features['test']
  features_train_hog = calculate_hog(features_train)
  features_test_hog = calculate_hog(features_test)
  np.savez_compressed(path,train=features_train_hog,test=features_test_hog)

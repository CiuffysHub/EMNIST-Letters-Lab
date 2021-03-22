import gzip
import numpy as np
import os
import sys

def load_mnist(kind,path):

	labels_path = os.path.join(path,
	                           '%s-labels-idx1-ubyte.gz'
	                           % kind)
	images_path = os.path.join(path,
	                           '%s-images-idx3-ubyte.gz'
	                           % kind)

	with gzip.open(labels_path, 'rb') as lbpath:
	    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
	                           offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
	    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
	                           offset=16).reshape(len(labels), 784)

	return images, labels


if __name__ == "__main__":

  features_train, labels_train = load_mnist('emnist-letters-train',sys.argv[1])
  features_test, labels_test = load_mnist('emnist-letters-test',sys.argv[1])
  np.savez_compressed('features_original',test=features_test,train=features_train)
  np.savez_compressed('labels_original',train=labels_train,test=labels_test)

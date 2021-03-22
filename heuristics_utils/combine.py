import numpy as np
import sys

if __name__ == "__main__":
	features1 = np.load(sys.argv[1])
	features1_train = features1['train']
	features1_test = features1['test']
	features2 = np.load(sys.argv[2])
	features2_train = features2['train']
	features2_test = features2['test']
	features_combined_train = [list(features1_train[i])+list(features2_train[i]) for i in range(0,len(features1_train))]
	features_combined_test = [list(features1_test[i])+list(features2_test[i]) for i in range(0,len(features1_test))]
	np.savez_compressed(sys.argv[3],train=features_combined_train,test=features_combined_test)


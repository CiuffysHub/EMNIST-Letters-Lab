from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
	features_file = sys.argv[2]
	labels_file = sys.argv[1]
	output_file = sys.argv[3]

	features = np.load(features_file)
	features_train = features['train']
	features_test = features['test']

	labels = np.load(labels_file)
	labels_train = labels['train']
	labels_test = labels['test']

	features_train_noL = [features_train[i] for i in range(0,len(features_train)) if labels_train[i]!=12]
	labels_train_noL = [labels_train[i] for i in range(0,len(features_train)) if labels_train[i]!=12]
	features_train_noI = [features_train[i] for i in range(0,len(features_train)) if labels_train[i]!=9]
	labels_train_noI = [labels_train[i] for i in range(0,len(features_train)) if labels_train[i]!=9]
	print('training RF no L...')
	rfc = RandomForestClassifier(n_jobs=-1, n_estimators=150)
	rfc.fit(features_train_noL, labels_train_noL)
	test_prediction_noL=rfc.predict(features_test)
	train_prediction_noL=rfc.predict(features_train)
	print('training RF no I...')
	rfc = RandomForestClassifier(n_jobs=-1, n_estimators=150)
	rfc.fit(features_train_noI, labels_train_noI)
	test_prediction_noI=rfc.predict(features_test)
	train_prediction_noI=rfc.predict(features_train)
	new_test_labels = [27 if (test_prediction_noL[i]==test_prediction_noI[i]-3 and test_prediction_noI[i]==12) else labels_test[i] for i in range(0,len(labels_test))]
	new_train_labels = [27 if (train_prediction_noL[i]==train_prediction_noI[i]-3 and train_prediction_noI[i]==12) else labels_train[i] for i in range(0,len(labels_train))]

	np.savez_compressed(output_file, test=np.asarray(new_test_labels),train=np.asarray(new_train_labels))

	'''for i in range(0,len(features_train)):
	  if new_train_labels[i]==27:
	    fig = plt.figure
	    plt.imshow(np.reshape(features_train[i],(28,28)),cmap='gray')
	    plt.show()'''
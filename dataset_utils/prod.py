import numpy as np
import matplotlib.pyplot as plt

f = np.load('features_original.npz')['train']
l = np.load('illabels.npz')['train']
o = np.load('labels_original.npz')['train']
c = 0
ci = 0
images = []
for i in range((len(f))):
	if l[i]==27:
		image = np.transpose(np.reshape(f[i],(28,28)))
		if c<300:
			images.append(image)
			if (o[i]==9):
				ci=ci+1
			c = c+1

print(ci)
images = np.reshape(images,(30,10,28,28))
row = np.zeros((280,28))
for a in range(0,30):
	line = np.zeros((28,28))
	for b in range(0,9):
		line = np.concatenate((line,images[a,b]))
	row = np.concatenate((row,line),axis = 1)
fig = plt.figure
plt.imshow(row,cmap='gray_r')
plt.show()
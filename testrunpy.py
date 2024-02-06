import numpy as np
from sklearn.datasets import fetch_california_housing
from build.release import _pb_mmmogpg as mgpg
import csv

def run():
	print("RunPY")
	#train_file = "data/train/airfoil_full_train.csv"
	#test_file = "data/test/airfoil_full_test.csv"
#testX = np.ndarray()
#testY = np.ndarray()
	#with open(train_file, newline='') as f:
	#	reader = csv.reader(f)
	#	trainX = np.mat(reader,dtype='float32')
	#	for row in reader:
	#		print(row)

	X, y = fetch_california_housing(return_X_y=True)
	print(X.shape)
	print(y.shape)
	mgpg.evolve("--file=params/py_params_multimodal.txt",X,y)

if __name__ == '__main__':
	run()

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
	#models = mgpg.evolve(X,y,file="params/py_params_multimodal.txt",seed=1)

	args_dict = {
		'time':-1,
		'generations':100, # Set to 1-3 for testing purposes
		'evaluations':-1,
		'prob':'multiobj',
		'multiobj':'symbreg_diversified',
		'functions':'+_-_*_p/',
		'erc':True,
		'gomea':True,
		'gomfos':'LT_i',
		'initmaxtreeheight':2,
		'syntuniqinit':2000,
		'popsize':500,
		'seed':1,
		'parallel':1,
		'nrtrees':2,
		'writeoutput':True,
		'outputdirectory':'output',
	}
	models = mgpg.evolve(X,y,**args_dict)

	print("Output models:")
	for m in models:
		print(m)

if __name__ == '__main__':
	run()

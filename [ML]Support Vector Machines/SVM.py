from sklearn import svm
import random
from numpy import linspace
import matplotlib.pyplot as plt
def load_data():
	train_data = []
	train_target = []
	test_data = []
	test_target = []
	indx_neg_test = random.sample(range(0, 5000), 2500)
	with open('negative.dat', 'r') as f:
		neg_data = f.readlines()
		pat_indx = 0
		for pat in neg_data:
			pat = pat.split(' ')
			pat_data = []
			for dim in pat:
				if dim.find(':')==-1:
					continue
				else:
					dim = dim[dim.find(':')+1:]
					pat_data.append(float(dim))

			if pat_indx in indx_neg_test:
				test_data.append(pat_data)
				test_target.append(-1)
			else:
				train_data.append(pat_data)
				train_target.append(-1)
			pat_indx += 1
	indx_pos_test = random.sample(range(0, 5000), 2500)
	with open('positive.dat', 'r') as f:
		pos_data = f.readlines()
		pat_indx = 0
		for pat in pos_data:
			pat = pat.split(' ')
			pat_data = []
			for dim in pat:
				if dim.find(':')==-1:
					continue
				else:
					dim = dim[dim.find(':')+1:]
					pat_data.append(float(dim))

			if pat_indx in indx_pos_test:
				test_data.append(pat_data)
				test_target.append(1)
			else:
				train_data.append(pat_data)
				train_target.append(1)
			pat_indx += 1
	return test_data,test_target,train_data,train_target

if __name__== '__main__':
	test_data,test_target,train_data,train_target = load_data()
	
	plt.figure(1)
	for training_size in map(int,linspace(1000,5000,5)):
		cur_train_data = []
		cur_train_target = []
		for i in range(training_size//2):
			cur_train_data.append(train_data[i])
			cur_train_target.append(train_target[i])
			cur_train_data.append(train_data[i+2500])
			cur_train_target.append(train_target[i+2500])

		# Gaussian Kernel Function (gamma=0.5/(50^2)?)
		gamma_range = linspace(1e-6,1e-2,20)
		error = []
		for g in gamma_range:
			clf = svm.SVC(kernel='rbf', gamma=g)
			clf.fit(cur_train_data,cur_train_target)
			error.append(1-clf.score(test_data,test_target))
			print('Training Size: ',training_size,' Gamma: ',g,
						' Error: ',1-clf.score(test_data,test_target))
		plt.subplot(3,2,training_size//1000)
		plt.plot(gamma_range,error)

	training_error = []
	testing_error = []
	sizes = linspace(1000,5000,5)
	gamma_min = 0.00157978947368

	for training_size in map(int,linspace(1000,5000,5)):
		clf = svm.SVC(kernel='rbf', gamma=gamma_min)
		cur_train_data = []
		cur_train_target = []
		for i in range(training_size//2):
			cur_train_data.append(train_data[i])
			cur_train_target.append(train_target[i])
			cur_train_data.append(train_data[i+2500])
			cur_train_target.append(train_target[i+2500])

		clf.fit(cur_train_data,cur_train_target)
		training_error.append(
									1-clf.score(cur_train_data,cur_train_target))
		testing_error.append(1-clf.score(test_data,test_target))
	plt.figure(2)
	plt.plot(sizes,training_error,'o-',color='r',label='train')
	plt.plot(sizes,testing_error,'o-',color='g',label='test')
	plt.legend(loc="best")
	print('Asymptotic Error for SVM is nearly 1.575.')
	print('Bayes classifier chooses the nearest mean to the sample.')
	bayes_target = []
	for pat in test_data:
		dis1 = sum([(x-1)*(x-1) for x in pat])
		dis2 = sum([(x+1)*(x+1) for x in pat])
		if(dis1<dis2):
			bayes_target.append(1)
		else:
			bayes_target.append(-1)
	diff = 0
	for pat in range(5000):
		if(bayes_target[pat]!=test_target[pat]):
			diff += 1
	print('Bayes Error for the 5000 patterns of the testing data is:',diff/5000)
	plt.show()

#Owen Howell, July 20, 2019
#olh20@bu.edu, https://owenhowell20.github.io

#This code runs Eco_SVM on MNIST dataset
#Note: This code takes significant computational time (+1 days aprox) , for the plots made in paper each realization was done in parallel
#Note: The memory requirments are also large for full dataset. For running on personal compute please subsample data
#Many thanks to https://www.bu.edu/tech/support/research/ for their advice on optimization

#Import standard python packages
import numpy as np 	
import matplotlib.pyplot as plt
import sys

#QP is done with CVXOPT packages
from cvxopt import matrix, solvers
import numpy as np
solvers.options['show_progress'] = False 


#A global error threshold, any small number
thresh = 1e-3


#Note: For each realization the C slack hyperparameter and gamma RBF hyperparameter should be tuned to minimize out of sample error


#it is approxmitly five for most realizations
C = 5.321

#using the 'auto' scikit learn SVM parameters
gamma = 1/(28*28)


#defining a RBF kernel function, tunable parameter sigma
def kernel(x,y):

	return np.exp( - gamma * np.dot( ( x - y ) , np.transpose(x - y) )  )

#Intilize the EcoSVM, compute support vectors for first N_start points
#Inputs are the datapoints, data label and Slack value
#Returns the set of active datapoints, active datapoint labels, support vector values and an active index value
def EcoSVM_initialize(xvals,yvals ):

	N_start = len(yvals)

	#Function to generate the intial kernel matrix
	def intial_kernel_matrix():

		#Compute the intial kernel matrix
		Qmat = np.zeros([N_start,N_start])

		for i in range(N_start):
			for j in range(N_start):

				#using a linear kernel
				s =  kernel(xvals[i,:],xvals[j,:])

				Qmat[i,j] = s*yvals[i]*yvals[j]

		return Qmat

	Qmat = intial_kernel_matrix()

	#Convert to CVXOPT format
	Q = matrix(Qmat)

	p = - np.ones(N_start)
	p = matrix(p)


	G = np.zeros([2*N_start,N_start])
	for i in range(N_start):
		G[i,i] = -1

	for i in range(N_start):
		G[i+N_start,i] = +1

	G = matrix(G)

	h = np.zeros([2*N_start])
	for i in range(N_start,2*N_start):
		h[i] = C
	h = matrix(h)

	A = np.zeros(N_start)
	for i in range(N_start):
		A[i] = yvals[i]

	A = matrix(A,(1,N_start),'d')

	b = matrix(0.0)

	sol =  solvers.qp(Q, p, G, h, A, b)

	#the intial values of solution
	#KKT values a_{i}
	KKT = np.array( sol['x'] )


	#only care about non-zero values
	for i in range(N_start):
		if (KKT[i] < thresh) :
			KKT[i] = 0.0

	#Only need to keep non-zero KKT values, also know as support vectors
	#Find intial support vector values and support vector indices
	support_vects_inds = np.array( np.ndarray.nonzero(KKT)[0] )
	support_vects = KKT[support_vects_inds]

	#the set of active datapoints
	active_data_x = intial_xvals[support_vects_inds,:]
	active_data_y = intial_yvals[support_vects_inds]

	#Check that there is at least one active support vector

	num_active = 0
	for i in range(len(support_vects_inds)):

		if ( (support_vects[i] - C)**2 > thresh  ):

			num_active = num_active + 1

	if (   num_active == 0 ):

		print("No active support vector found. Make sure that there are both +1 and -1 examples. Increase the number of intial points. Increase the slack.")
		quit()


	#Find the active index
	test_vals = (support_vects - C/2.0 )**2
	index_val = np.argmin(test_vals)

			
	return active_data_x, active_data_y, support_vects, index_val


#Run the EcoSVM algorithm on a single new point
#Inputs are datapoint X, datalabel Y, active datapoints, active data labels, set of support vectors Lagrange Multiplier, dataset dimension and Slack value
def point_Run_EcoSVM( X, Y , active_data_x , active_data_y , support_vects , index_val , dimension ):

	numsupportvects = len(active_data_y)
	
	#Find the active index

	test_vals = (support_vects - C/2.0 )**2
	index_val = np.argmin(test_vals)


	s = 0

	for i in range(numsupportvects):

		s = s + active_data_y[i]*Y*( kernel( active_data_x[i,:], active_data_x[index_val,:]  ) -  kernel( active_data_x[i,:], X  )       )


	#Compute the invasion condition
	inv = 1 - Y*active_data_y[index_val] + s

	if (inv>=0):


		#The new species can invade. Recompute the steady state using QP

		Qp = np.zeros([numsupportvects+1,numsupportvects+1])


		for i in range(numsupportvects):
			for j in range(numsupportvects):

					s = kernel(active_data_x[i,:],active_data_x[j,:])

					Qp[i,j] = s*active_data_y[i]*active_data_y[j]

		for i in range(numsupportvects):

			s = kernel(active_data_x[i,:], X)

			Qp[i,numsupportvects] = s*active_data_y[i]*Y

			Qp[numsupportvects,i] = s*active_data_y[i]*Y
		

		s = kernel(X,X)
	
		Qp[numsupportvects,numsupportvects] = s*Y * Y

		Qp = matrix(Qp)

		p = - np.ones(numsupportvects+1)
		p = matrix(p)

		G = np.zeros([2*numsupportvects+2,numsupportvects+1])

		for i in range(numsupportvects+1):
			G[i,i] = -1

		for i in range(numsupportvects+1):

			G[i+numsupportvects+1,i] = +1


		G = matrix(G)


		h = np.zeros([2*numsupportvects+2])
		for i in range(numsupportvects+1,2*numsupportvects+2):
			h[i] = C
		h = matrix(h)

		A = np.zeros(numsupportvects+1)
		for i in range(numsupportvects):
			A[i] = active_data_y[i]

		A[numsupportvects] = Y
		A = matrix(A,(1,numsupportvects+1),'d')

		b = matrix(0.0)

		#Call QP function
		sol =  solvers.qp(Qp, p, G, h, A, b)

		#QP solution as array, all KKT values
		KKT = np.array( sol['x'] )


		#Get the new support vector indices and values
		#only care about non-zero support vectors
		countnew = 0
		for i in range(len(KKT)):
			if (KKT[i] < thresh):
				KKT[i] = 0
				countnew = countnew + 1

		countnew = len(KKT) -  countnew


		#the set of new support vectors and support vector indices
		new_active_data_x = np.zeros([countnew, dimension])
		new_active_data_y = np.zeros([countnew])
		newsuppvects = np.zeros([countnew])


		auxcount = 0
		auxcount2 = 0
		for i in range(len(KKT)-1):

			if (KKT[i] > thresh):
				new_active_data_x[auxcount,:] = active_data_x[auxcount2,:]
				new_active_data_y[auxcount] = active_data_y[auxcount2]
				auxcount2 = auxcount2 + 1
		
				newsuppvects[auxcount] = KKT[i]
				auxcount = auxcount + 1
				
			if (KKT[i]<thresh):
				auxcount2 = auxcount2 + 1


		if (KKT[len(KKT)-1] > thresh):


			new_active_data_x[auxcount,:] = X
			new_active_data_y[auxcount] = Y
			newsuppvects[auxcount] = KKT[len(KKT)-1]

			auxcount = auxcount + 1


		#New support vector values and indices
		support_vects = newsuppvects

		active_data_y = new_active_data_y

		#zero array because it can change shape
		active_data_x = np.zeros( [len(support_vects) , dimension ] )
		active_data_x = new_active_data_x

	
	return active_data_x, active_data_y, support_vects, index_val


#Run the EcoSVM algorithm
#Inputs are datapoints and labels, set of intial support vector indices, intial support vector values, intial Lagrange Multiplier and Slack Value
#Returns the set of active datapoints, the active data labels, the support vector values and the final lagrange multiplier
def Run_EcoSVM( xvals, yvals, active_data_x, active_data_y, support_vects, index_val ):

	N = len(yvals)

	test_accuracy = np.zeros([ N - N_start])
	number_active = np.zeros([ N - N_start])

	#the dataset dimension
	dimension = len(xvals[0,:])


	#Run the EcoSVM algorithm over all points
	for point in range(N_start,N):

		#compute the b value
		b = b_value(active_data_x,active_data_y,support_vects)

		#Compute performance errors
		EcoSVMerror = SVM_error(test_xvals,test_yvals, active_data_x, active_data_y, support_vects,b)

		test_accuracy[point - N_start] = 1 - EcoSVMerror
		print( 1 - EcoSVMerror )

		count_active = 0 

		for i in range(len(active_data_y)):

			if (   support_vects[i] > thresh  and ( support_vects[i] - C )**2  > thresh**2 ):

				count_active = count_active + 1

		number_active[ point - N_start ] = count_active


		X = xvals[point  ,:]

		Y = yvals[point ]

		#Run the EcoSVM algorithm on a single point
		active_data_x, active_data_y, support_vects, index_val = point_Run_EcoSVM( X , Y , active_data_x, active_data_y , support_vects , index_val , dimension )

	
	return active_data_x, active_data_y , support_vects , index_val , test_accuracy, number_active


#Run a batch SVM on all data
#input is all training data, training labels and Slack value
#output is the set of active datapoints and data labels and support vector values
def batchSVM( xvals , yvals ):

	#the number of datapoints
	N = len(yvals)

	#the full kernel matrix for batch SVM
	Qfull = np.zeros([N,N])

	for i in range(N):
		for j in range(N):

			#using a linear kernel
			s =  kernel(xvals[i,:],xvals[j,:])

			Qfull[i,j] = s*yvals[i]*yvals[j]

	#The full batch SVM solution with QP	
	#Convert into CVXOPT format

	Qf = matrix(Qfull)

	pf = - np.ones(N)
	pf = matrix(pf)

	Gf = np.zeros([2*N,N])
	for i in range(N):
		Gf[i,i] = -1

	for i in range(N):
		Gf[N+i,i] = +1


	Gf = matrix(Gf)

	hf = np.zeros([2*N])
	for i in range(N,2*N):
		hf[i] = C

	hf = matrix(hf)

	Af = np.zeros(N)
	for i in range(N):
		Af[i] = yvals[i]

	Af = matrix(Af,(1,N),'d')

	bf = matrix(0.0)

	sol =  solvers.qp(Qf, pf, Gf, hf, Af, bf)

	evars = np.array( sol['x'] )


	#only care about non-zero support vectors
	for i in range(N):
		if (evars[i] < thresh):
			evars[i] = 0.0


	#Find support vectors and support vector indices for Batch SVM
	supvectsindsfull = np.array( np.ndarray.nonzero(evars)[0] )
	supvectsfull = evars[supvectsindsfull]

	active_data_x = xvals[ supvectsindsfull , :]
	active_data_y = yvals[supvectsindsfull ]

	return active_data_x, active_data_y, supvectsfull


#Compute the B value for an SVM
#Inputs are indices and support vector values
def b_value(active_data_x, active_data_y,supportvectors ):

	s = 0 
	bp = 0

	for i in range(len(supportvectors)):

		bp = bp + 1

		s = s + active_data_y[i]

		for j in range(len(supportvectors)):

			s = s - supportvectors[j] * active_data_y[j] * kernel( active_data_x[i,:] ,  active_data_x[j,:] )
			

	b = 0
	if (bp!=0):
		b = 1/float(bp) * s

	return b


#the SVM prediction function
#Inputs are datapoint x to make prediction on, set of indices, set of support vectors and b value
#Output is the prediction value +1 or -1
def pred(x , active_data_x, active_data_y , supportvectors , b):
	
	s = 0

	for i in range(len(supportvectors)):

		s = s + active_data_y[i] * supportvectors[i] * kernel(x , active_data_x[i,:]  )

	s = s + b


	return s


#Function to compute test error
#Inputs are testing data and labels, set of support vector indices and support vector values
#Returns test error
def SVM_error( test_xvals , test_yvals , active_data_x, active_data_y , support_vects, b):

	#the number of test points
	N_test = len( test_yvals )

	#Compute the EcoSVM error, # of missclassified points
	error = 0
	for i in range(N_test):


		if ( test_yvals[i] != np.sign( pred( test_xvals[i] , active_data_x, active_data_y, support_vects, b ) )  ):

			error = error + 1


	return error/N_test



#The MNIST dataset, https://ci.nii.ac.jp/naid/10027939599/en/ for more information
from keras.datasets import mnist
#each image is 28x28 , dimension of each image
dimension = 28 * 28


#Function to get MNIST dataset
def getMNIST():

	(all_xvals, all_yvals), (all_test_xvals , all_test_yvals) = mnist.load_data()


	#reshape data into useable form
	all_xvals = np.reshape(all_xvals,(60000,28*28))
	all_test_xvals = np.reshape(all_test_xvals,(10000,28*28))


	#Count the number of digits:
	countA = 0
	countB = 0
	for i in range( len(all_yvals)  ):

		if ( all_yvals[i] == 1 ):

			countA = countA + 1

		if ( all_yvals[i] == 4 ):

			countB = countB + 1


	xvals = np.zeros([countA+countB, 28*28])
	yvals = np.zeros([countA + countB])


	count = 0
	for i in range( len(all_yvals)  ):

		if ( all_yvals[i] == 1 ):

			xvals[count,:] = all_xvals[i,:]
			yvals[count] = +1
			count = count + 1


		if ( all_yvals[i] == 4 ):

			xvals[count,:] = all_xvals[i,:]
			yvals[count] = -1
			count = count + 1


	countA_test = 0
	countB_test = 0
	for i in range( len(all_test_yvals)  ):

		if ( all_test_yvals[i] == 1 ):

			countA_test = countA_test + 1

		if ( all_test_yvals[i] == 4 ):

			countB_test = countB_test + 1


	test_xvals = np.zeros([countA_test+countB_test, 28*28])
	test_yvals = np.zeros([countA_test + countB_test])


	count = 0
	for i in range( len(all_test_yvals)  ):

		if ( all_test_yvals[i] == 1 ):

			test_xvals[count,:] = all_test_xvals[i,:]
			test_yvals[count] = +1
			count = count + 1


		if ( all_test_yvals[i] == 4 ):

			test_xvals[count,:] = all_test_xvals[i,:]
			test_yvals[count] = -1
			count = count + 1


	#Essential to rescale data as MNIST set has rank less than dimension
	from sklearn import preprocessing
	scaler = preprocessing.StandardScaler().fit(xvals)

	xvals = scaler.transform( xvals )
	test_xvals = scaler.transform( test_xvals )

	return xvals, yvals, test_xvals, test_yvals


xvals, yvals, test_xvals, test_yvals = getMNIST()

#Shuffle the order of the training values
from sklearn.utils import shuffle
xvals , yvals = shuffle( xvals , yvals, random_state=0)


#the labels
yvals = np.array(yvals)
test_yvals = np.array(test_yvals)


#Total number of training points
N = len(yvals)

#Total number of test points
N_test = len(test_yvals)

#Intial number of points used to compute steady state, can be user entered
#This should be much greater than dataset dimension especily if dataset is highly non-linear
#For MNIST dataset with RBF kernel the data set is essentialy linear so there is no problem with using small number of points
N_start = 10

#the intial datapoints and labels
intial_xvals = xvals[0:N_start,:]
intial_yvals = yvals[0:N_start]


#subsample to run in reasonable time
#for results in paper please use whole dataset
Ntrun = 100

xvals = xvals[0:Ntrun,:]
yvals = yvals[0:Ntrun]


#Get the intial set of active datapoints, active datapoint labels, support vector values and the Lagrange multiplier
intial_active_data_x, intial_active_data_y, intial_support_vects , intial_index = EcoSVM_initialize(intial_xvals,intial_yvals)


#Run the EcoSVM algorithm on the dataset
active_data_x, active_data_y , support_vects, index_val,  test_accuracy, number_active = Run_EcoSVM( xvals, yvals, intial_active_data_x, intial_active_data_y, intial_support_vects , intial_index)


#Get the full batch solution to compare
batch_data_x, batch_data_y,  batch_support_vects = batchSVM( xvals,yvals)
batch_number_active =  len(batch_data_y)

#compute the batch b value
bfull = b_value(batch_data_x,batch_data_y,batch_support_vects)
#compute the batch accuracy
batcherror = SVM_error(test_xvals,test_yvals, batch_data_x, batch_data_y, batch_support_vects,bfull)


#averge batch error, averge number of batch support vectors
batcherror = batcherror
batch_number_active = batch_number_active

#make accuracy plots vs time
import os

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)

fontsize = 22


plt.plot( test_accuracy , linewidth=2, color = 'k')

plt.axhline(y = 1 - batcherror,linestyle='--',linewidth=6)
plt.ylim(0.5,1.1)
plt.ylabel("$A(T) $",fontsize = fontsize + 2)
plt.xlabel("$ T  $" ,fontsize=fontsize + 2)
plt.grid()
plt.tick_params(labelsize=fontsize + 2)
plt.tight_layout()
plt.show()
plt.savefig("./graphs/mnistacc")
plt.clf()


plt.plot( number_active , linewidth=2 , color = 'k')
plt.axhline(y = batch_number_active,linestyle='--',linewidth=6)
plt.ylabel("$N(T) $",fontsize = fontsize + 2)
plt.xlabel("$ T  $" ,fontsize=fontsize + 2)
plt.tick_params(labelsize=fontsize + 2)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig("./graphs/mnistnum")
plt.clf()






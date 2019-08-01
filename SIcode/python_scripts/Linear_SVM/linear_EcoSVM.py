#Owen Howell, July 15, 2019
#olh20@bu.edu, https://owenhowell20.github.io

#Optimized linear EcoSVM code
#Nothing is precomputed

#This code runs EcoSVM algoritm and compares with batch SVM

#Import standard python packages
import numpy as np 	
import matplotlib.pyplot as plt
import sys

#QP is done with CVXOPT packages
from cvxopt import matrix, solvers
import numpy as np
solvers.options['show_progress'] = False 


#A global error threshold, any small number can be used
thresh = 1e-3

#Function to generate the training data. Any linearly seperable dataset can be used.
#Returns dataset and dateset labels
def train_data(N, dimension):
	
	#Draw set of random points
	xvals = np.random.uniform(0,1,[N,dimension])

	yvals = np.ones([N])
	for i in range(N):
		
		#Linearly Seperable
		if (xvals[i,0]>0.5):

			yvals[i] = -1



	return xvals, yvals

#Function to generate the test data. Any linearly seperable dataset can be used.
#Returns dataset and dateset labels
def test_data(N_test,dimension):
	
	#Draw set of random points
	xvals = np.random.uniform(0,1,[N_test,dimension])

	yvals = np.ones([N_test])
	for i in range(N_test):
		
		#Linearly Seperable
		if (xvals[i,0] > 0.5):

			yvals[i] = -1



	return xvals, yvals

#Defining a linear kernel function
def kernel(x,y):

	return np.dot(x,np.transpose(y))

#Intilize the EcoSVM, compute support vectors for first N_start points
#Inputs are the datapoints, data labels
#Returns the set of active datapoints, active datapoint labels, support vector values
def EcoSVM_initialize(xvals,yvals):

	N_start = len(yvals)
 
	#Function to generate the intial kernel matrix
	def intial_kernel_matrix():

		#Compute the intial kernel matrix
		Qmat = np.zeros([N_start,N_start])

		for i in range(N_start):
			for j in range(N_start):

				#using a linear kernel
				s =  kernel(xvals[i],xvals[j])

				Qmat[i,j] = s*yvals[i]*yvals[j]

		return Qmat

	Qmat = intial_kernel_matrix()

	#Convert to CVXOPT format
	Q = matrix(Qmat)

	p = - np.ones(N_start)
	p = matrix(p)

	G = np.zeros([N_start,N_start])
	for i in range(N_start):
		G[i,i] = -1

	G = matrix(G)

	h = np.zeros([N_start])
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
			KKT[i] = 0

	#Only need to keep non-zero KKT values, also know as support vectors
	#Find intial support vector values and support vector indices
	support_vects_inds = np.array( np.ndarray.nonzero(KKT)[0] )
	support_vects = KKT[support_vects_inds]

	#the set of active datapoints
	active_data_x = intial_xvals[support_vects_inds,:]
	active_data_y = intial_yvals[support_vects_inds]

	#Check that there is at least one active support vector

	if (   len(support_vects_inds) == 0 ):

		print("Not enough intial points, no active support vector found. Make sure that there are both +1 and -1 examples.")
		quit()


	return active_data_x, active_data_y, support_vects


#Run the EcoSVM algorithm on a single new point
#Inputs are datapoint X, datalabel Y, active datapoints, active data labels, set of support vectors and the dataset dimension
#Returns the new set of data points and labels, the new set of support vectors
def point_Run_EcoSVM( X, Y , active_data_x , active_data_y , support_vects , dimension  ):


	numsupportvects = len(active_data_y)

	s = 0


	for i in range(numsupportvects):

		Qval = Y*active_data_y[i]*(  kernel( X , active_data_x[i,:]  )  - kernel( active_data_x[0,:], active_data_x[i,:]  ) )

		s = s + Qval*support_vects[i]
	

	#Compute the invasion condition
	inv = 1 - Y * active_data_y[0] - s

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


		G = np.zeros([numsupportvects+1,numsupportvects+1])
		for i in range(numsupportvects+1):
			G[i,i] = -1
		G = matrix(G)


		h = np.zeros([numsupportvects+1])
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
		

		# #set up some indices to check
		# indexarray = np.zeros([numsupportvects+1])
		# for i in range(numsupportvects):
		# 	indexarray[i] = support_vects_inds[i]

		# indexarray[numsupportvects] = point

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

		
				newsuppvects[auxcount] = KKT[i]
				auxcount = auxcount + 1
				auxcount2 = auxcount2 + 1

			if (KKT[i]<thresh):
				auxcount2 = auxcount2 + 1



		if (KKT[len(KKT)-1]>thresh):


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

	
	return active_data_x, active_data_y, support_vects


#Run the EcoSVM algorithm
#Inputs are datapoints and labels, set of intial support vector indices, intial support vector values
#Returns the set of active datapoints, the active data labels, the support vector values
def Run_EcoSVM( xvals, yvals, active_data_x, active_data_y, support_vects ):

	#the dataset dimension
	dimension = len(xvals[0,:])

	#Run the EcoSVM algorithm over all points
	for point in range(N_start,N):

		X = xvals[point,:]
		Y = yvals[point]

		#Run the EcoSVM algorithm on a single point
		active_data_x, active_data_y, support_vects = point_Run_EcoSVM( X , Y , active_data_x, active_data_y , support_vects , dimension )

	
	return active_data_x, active_data_y , support_vects

#Run a batch SVM on all data
#input is all training data and training labels
#output is the set of active datapoints and data labels and support vector values
def batchSVM( xvals , yvals ):

	#the number of datapoints
	N = len(yvals)

	#the full kernel matrix for batch SVM
	Qfull = np.zeros([N,N])

	for i in range(N):
		for j in range(N):

			#using a linear kernel
			s =  kernel(xvals[i],xvals[j])

			Qfull[i,j] = s*yvals[i]*yvals[j]

	#The full batch SVM solution with QP	
	#Convert into CVXOPT format

	Qf = matrix(Qfull)

	pf = - np.ones(N)
	pf = matrix(pf)


	Gf = np.zeros([N,N])
	for i in range(N):
		Gf[i,i] = -1

	Gf = matrix(Gf)

	hf = np.zeros([N])
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
			evars[i] = 0


	#Find support vectors and support vector indices for Batch SVM
	supvectsindsfull = np.array( np.ndarray.nonzero(evars)[0] )
	supvectsfull = evars[supvectsindsfull]

	active_data_x = xvals[ supvectsindsfull ,:]
	active_data_y = yvals[supvectsindsfull ]

	return active_data_x, active_data_y, supvectsfull


#Compute the b value for an SVM
#Inputs are set of active datapoints and data labels and support vector values
def b_value(active_data_x,active_data_y,supportvectors):

	b = 0
	#Compute the b value
	s = 0 
	for i in range(len(supportvectors)):

		s = s + active_data_y[i]

		for j in range(len(supportvectors)):

			s = s - supportvectors[j] * active_data_y[j] * kernel( active_data_x[i,:] , active_data_x[j,:]     )


	size = float( len(supportvectors) )
	if (size!=0):
		b = 1/size * s

	if (size==0):
		print("ERROR")

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
def SVM_error( test_xvals , test_yvals , active_data_x, active_data_y , support_vects,b):

	#the number of test points
	N_test = len( test_yvals )

	#Compute the EcoSVM error, # of missclassified points
	error = 0
	for i in range(N_test):


		if ( test_yvals[i] != np.sign( pred( test_xvals[i] , active_data_x, active_data_y, support_vects, b ) )  ):

			error = error + 1


	return error/N_test


#These parameters are chosen to create a dataset
#Dimension of the dataset
dimension = 2

#Total number of training points
N = 200

#Get train and test datasets, this can be user entered
xvals, yvals = train_data(N,dimension)

#Total number of test points
N_test = 1000

test_xvals , test_yvals = test_data(N_test,dimension)


#Intial number of points used to compute steady state, can be user entered
N_start = 10

#the intial datapoints and labels
intial_xvals = xvals[0:N_start,:]
intial_yvals = yvals[0:N_start]


#Get the intial set of active datapoints, active datapoint labels, support vector values and the Lagrange multiplier
intial_active_data_x, intial_active_data_y, intial_support_vects = EcoSVM_initialize(intial_xvals,intial_yvals)

#Run the EcoSVM algorithm on the dataset
active_data_x, active_data_y , support_vects = Run_EcoSVM( xvals, yvals, intial_active_data_x, intial_active_data_y, intial_support_vects  )

#Get the full batch solution to compare
batch_data_x, batch_data_y,  batch_support_vects = batchSVM( xvals,yvals )


#compute the b value
bfull = b_value(batch_data_x,batch_data_y,batch_support_vects)
b = b_value(active_data_x,active_data_y,support_vects)


#Compute performance errors
EcoSVMerror = SVM_error(xvals,yvals, active_data_x, active_data_y, support_vects,b)
print('EcoSVM train error:',EcoSVMerror)
EcoSVMerror = SVM_error(test_xvals,test_yvals, active_data_x, active_data_y, support_vects,b)
print('EcoSVM test error:',EcoSVMerror)

print('')

batcherror = SVM_error(xvals,yvals, batch_data_x, batch_data_y , batch_support_vects,bfull)
print('Batch SVM train error:', batcherror)
batcherror = SVM_error(test_xvals,test_yvals, batch_data_x, batch_data_y, batch_support_vects,bfull)
print('Batch SVM test error:', batcherror)


#Function to make prediction plots
#See main text for detail
#only make plots in two dimensions
def make_plot():

	#Only make plots for two dimensional data
	if (dimension!=2):
		quit()


	#Make prediction plots
	k = 200
	batch_preds = np.zeros([k,k])
	EcoSVM_preds = np.zeros([k,k])
	x = 0
	y = 0
	dl = 1/k

	for i in range(k):

		x = 0

		for j in range(k):

			batch_preds[i,j] = np.sign( pred([x,y],batch_data_x,batch_data_y, batch_support_vects,bfull) )

			EcoSVM_preds[i,j] = np.sign( pred( [x,y] , active_data_x, active_data_y, support_vects,b ) )

			x = x + dl

		y = y + dl


	diffs = 0*batch_preds

	x = 0
	y = 0
	dl = 1/k

	for i in range(k):

		x = 0

		for j in range(k):

			if (batch_preds[i,j] == EcoSVM_preds[i,j]):

				if (EcoSVM_preds[i,j]==1):

					diffs[i,j] = -1

				if (EcoSVM_preds[i,j]==-1):

					diffs[i,j] = 1



			if (batch_preds[i,j]!=EcoSVM_preds[i,j]):

				diffs[i,j] = 3

			x = x + dl

		y = y + dl


	#plot all train datapoints
	for i in range(N):

		if (yvals[i]==1):
			plt.plot(xvals[i,0],xvals[i,1],'.',c='g',marker='P',markersize=8,markeredgecolor='black')

		if (yvals[i]==-1):
			plt.plot(xvals[i,0],xvals[i,1],'.',c='r',marker='o',markersize=8,markeredgecolor='black')


	#also plot active support vectors in larger markers
	for i in range(len(support_vects)):


		if (active_data_y[i]==1):

			plt.plot(active_data_x[i,0],active_data_x[i,1],'.',c='g',marker='P',markersize=22,markeredgecolor='black')

		if (active_data_y[i]==-1):

			plt.plot(active_data_x[i,0], active_data_x[i,1],'.',c='r',marker='o',markersize=22,markeredgecolor='black')

	fontsize = 20
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.grid()
	plt.tick_params(labelsize=fontsize)
	plt.xlabel("$X_{1}$",size=fontsize+2)
	plt.ylabel("$X_{2}$",size = fontsize+2)


	from pylab import rcParams
	rcParams['figure.figsize'] = 500, 500

	plt.axvline(x=0.5,linestyle='--',linewidth=5,color='k')
	plt.imshow(diffs,origin='lower',extent=(0,1,0,1),cmap='cool')
	plt.tight_layout()
	plt.show()
	

make_plot()



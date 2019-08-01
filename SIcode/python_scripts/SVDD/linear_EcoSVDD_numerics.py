#Owen Howell, July 15, 2019
#olh20@bu.edu, https://owenhowell20.github.io

#Optimized linear EcoSVDD code
#In case of norm 1 kernel K( x, x) = 1 reduces to FISVDD algorithm: https://arxiv.org/abs/1709.00139
#In paper we focus on kernel functions K(x,x) = 1, however this code works for any kernel function K(x,y)

#This code runs EcoSVDD algoritm and compares with batch SVDD
#Compares accuracy vs time

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


#Generate data from a gaussian distribution
#Returns dataset
def generate_data(N, dimension):


	#The mean of the distribtuion
	mean  = np.random.uniform( 0.5, 0.5, dimension)

	#The covarience matrix of the distribution

	cov = np.zeros([dimension,dimension])

	for i in range(dimension):
		cov[i,i] = .01

	xvals = np.random.multivariate_normal(mean,cov,N)

	#xvals = np.random.uniform(0.2,0.6,[N,dimension])

	return xvals


#The kernel function
def kernel(x,y):

	return np.dot(x,np.transpose(y))  / np.sqrt(  np.dot(x,np.transpose(x)) * np.dot(y,np.transpose(y))   )


#Intilize the EcoSVDD, compute support vectors for first N_start points
#Inputs are the datapoints
#Returns the set of active datapoints, support vector values
def EcoSVDD_initialize(xvals):

	N_start = len(xvals[:,0])
 
	#Function to generate the intial kernel matrix
	def intial_kernel_matrix():

		#Compute the intial kernel matrix
		Qmat = np.zeros([N_start,N_start])

		for i in range(N_start):
			for j in range(N_start):

				
				s =  kernel(xvals[i,:],xvals[j,:])

				Qmat[i,j] = 2.0*s

		return Qmat

	Qmat = intial_kernel_matrix()

	#Convert to CVXOPT format
	Q = matrix(Qmat)

	p = np.zeros(N_start)
	for i in range(N_start):
		p[i] = - kernel(xvals[i,:],xvals[i,:])

	p = matrix(p)

	G = np.zeros([N_start,N_start])
	for i in range(N_start):
		G[i,i] = -1

	G = matrix(G)

	h = np.zeros([N_start])
	h = matrix(h)

	A = np.zeros(N_start)
	for i in range(N_start):
		A[i] = 1

	A = matrix(A,(1,N_start),'d')

	b = matrix(1.0)

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

	#Check that there is at least one active support vector

	if (   len(support_vects_inds) == 0 ):

		print("Not enough intial points, no active support vector found. Make sure adaquate kernel function is used")
		quit()


	return active_data_x, support_vects


#Run the EcoSVDD algorithm on a single new point
#Inputs are datapoint X,  active datapoints, set of support vectors and the dataset dimension
#Returns the new set of data points and the new set of support vectors
def point_Run_EcoSVDD( X, active_data_x , support_vects , dimension  ):

	
	numsupportvects = len(active_data_x[:,0])

	#sum varible
	s = 0
	
	for i in range(numsupportvects):

		Qval = kernel( X , active_data_x[i,:]  ) - kernel( active_data_x[0,:] , active_data_x[i,:]  )

		s = s + Qval*support_vects[i]
	

	#Compute the invasion condition
	inv = kernel( X, X  ) - kernel( active_data_x[0,:] , active_data_x[0,:]  )  - s

	if (inv>=0):

		#The new species can invade. Recompute the steady state using QP

		Qp = np.zeros([numsupportvects+1,numsupportvects+1])


		for i in range(numsupportvects):
			for j in range(numsupportvects):

					s = kernel(active_data_x[i,:],active_data_x[j,:])

					Qp[i,j] = 2*s

		for i in range(numsupportvects):

			s = kernel(active_data_x[i,:], X)

			Qp[i,numsupportvects] = 2*s

			Qp[numsupportvects,i] = 2*s
		

		s = kernel(X,X)
	
		Qp[numsupportvects,numsupportvects] = 2*s

		Qp = matrix(Qp)

		p = np.zeros(numsupportvects+1)

		for i in range(numsupportvects):
			p[i] = - kernel(active_data_x[i,:],active_data_x[i,:])


		p[numsupportvects] = - kernel(  X, X  )


		p = matrix(p)


		G = np.zeros([numsupportvects+1,numsupportvects+1])
		for i in range(numsupportvects+1):
			G[i,i] = -1
		G = matrix(G)


		h = np.zeros([numsupportvects+1])
		h = matrix(h)


		A = np.zeros(numsupportvects+1)
		for i in range(numsupportvects):
			A[i] = 1

		A[numsupportvects] = 1
		A = matrix(A,(1,numsupportvects+1),'d')

		b = matrix(1.0)

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
		newsuppvects = np.zeros([countnew])


		auxcount = 0
		auxcount2 = 0
		for i in range(len(KKT)-1):

			if (KKT[i] > thresh):
				new_active_data_x[auxcount,:] = active_data_x[auxcount2,:]				

				newsuppvects[auxcount] = KKT[i]
				auxcount = auxcount + 1
				auxcount2 = auxcount2 + 1

			if (KKT[i]<thresh):
				auxcount2 = auxcount2 + 1



		if (KKT[len(KKT)-1]>thresh):

			new_active_data_x[auxcount,:] = X

			newsuppvects[auxcount] = KKT[len(KKT)-1]
			auxcount = auxcount + 1

		
		#New support vector values and indices
		support_vects = newsuppvects

		#zero array because it can change shape
		active_data_x = np.zeros( [len(support_vects) , dimension ] )
		active_data_x = new_active_data_x

	
	return active_data_x, support_vects


#Run the EcoSVM algorithm
#Inputs are datapoints and intial support vector values
#Returns the set of active datapoints, the support vector values
def Run_EcoSVDD( xvals, active_data_x, support_vects  ):


	radii = np.zeros([ N - N_start ])
	overlap = np.zeros([ N - N_start ])

	#the dataset dimension
	dimension = len(xvals[0,:])

	#Run the EcoSVM algorithm over all points
	for point in range(N_start,N):


		radii[ point - N_start ] = get_radius(active_data_x, support_vects)

		overlap[ point - N_start] = sim_metric( active_data_x, support_vects, batch_data_x, batch_support_vects   )

		X = xvals[point,:]

		#Run the EcoSVM algorithm on a single point
		active_data_x, support_vects = point_Run_EcoSVDD( X , active_data_x , support_vects, dimension )

	
	return active_data_x, support_vects , radii, overlap


#Run a batch SVDD on all data
#Input is all training data
#Output is the set of active datapoints and support vector values
def batchSVDD( xvals  ):

	#the number of datapoints
	N = len(xvals[:,0])

	#the full kernel matrix for batch SVM
	Qfull = np.zeros([N,N])

	for i in range(N):
		for j in range(N):

			#using a linear kernel
			s =  kernel(xvals[i],xvals[j])

			Qfull[i,j] = 2*s

	#The full batch SVM solution with QP	
	#Convert into CVXOPT format

	Qf = matrix(Qfull)

	pf = np.zeros(N)

	for i in range(N):
		pf[i] = -kernel(xvals[i,:],xvals[i,:])

	pf = matrix(pf)


	Gf = np.zeros([N,N])
	for i in range(N):
		Gf[i,i] = -1

	Gf = matrix(Gf)

	hf = np.zeros([N])
	hf = matrix(hf)

	Af = np.zeros(N)
	for i in range(N):
		Af[i] = 1

	Af = matrix(Af,(1,N),'d')

	bf = matrix(1.0)

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
	
	return active_data_x, supvectsfull


#Function to get the radius of the trained SVDD
#Inputs are active data and set of support vectors
#Returns the SVDD radius
def get_radius(active_data_x,support_vects):

	dists = np.zeros([len(support_vects)])

	for i in range(len(support_vects)):

		d1 = 0
		d2 = 0	

		for j in range(len(support_vects)):

			d1  = d1 + kernel( active_data_x[i,:] , active_data_x[j,:]  ) * support_vects[j]
		
			for k in range(len(support_vects)):

				d2 = d2 + kernel( active_data_x[j,:] , active_data_x[k,:]  ) * support_vects[j] * support_vects[k]



		dists[i] = kernel( active_data_x[i,:] , active_data_x[i,:]  ) - 2*d1  + d2


		R = np.sqrt(  min(dists)   )

	return R


#A similiarity metric between two SVDDs
#Returns the normilized dot product of the two SVDD sphere centers
def sim_metric( active_data_x, support_vects, batch_data_x, batch_support_vects  ):

	lenth_1 = len(support_vects)
	lenth_2 = len(batch_support_vects)

	val = 0

	for i in range(lenth_1):
		for j in range(lenth_2):

			val = val + support_vects[i]*batch_support_vects[j] * kernel( active_data_x[i,:] , batch_data_x[j,:]  )


	norm1 = 0
	for i in range(lenth_1):
		for j in range(lenth_1):

			norm1 = norm1 + support_vects[i]*support_vects[j] * kernel( active_data_x[i,:], active_data_x[j,:]  )


	norm2 = 0
	for i in range(lenth_2):
		for j in range(lenth_2):

			norm2 = norm2 + batch_support_vects[i]*batch_support_vects[j] * kernel( batch_data_x[i,:], batch_data_x[j,:]  )

	
	metric = val / np.sqrt(  norm1 * norm2 ) 
	metric = np.float(metric)
	return metric



#These parameters are chosen to create a dataset
#Dimension of the dataset
dimension = 100

#Total number of training points
N = 600

#Intial number of points used to compute steady state, can be user entered
N_start = 20

#Number of realizations
N_reals = 20

#The realization accuracies
radii_data = np.zeros([N_reals,N - N_start])
overlap_data = np.zeros([N_reals,N - N_start])

avg_batch_radius = 0
#loop over realizations
for r in range(N_reals):

	#get the training set
	xvals = generate_data(N,dimension)


	#Get the full batch solution to compare
	batch_data_x,  batch_support_vects = batchSVDD( xvals )

	#The batch SVDD radius
	batch_radius = get_radius( batch_data_x, batch_support_vects  )

	avg_batch_radius = avg_batch_radius + batch_radius

	#the intial datapoints and labels
	intial_xvals = xvals[0:N_start,:]


	#Get the intial set of active datapoints, support vector values and the Lagrange multiplier
	intial_active_data_x, intial_support_vects = EcoSVDD_initialize(intial_xvals)

	#Run the EcoSVM algorithm on the dataset
	active_data_x, support_vects, radii, overlap = Run_EcoSVDD( xvals, intial_active_data_x, intial_support_vects   )

	radii_data[r,:] = radii
	overlap_data[r,:] = overlap




#averge batch radius
avg_batch_radius = avg_batch_radius/N_reals

#make accuracy plots vs time
import os

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)

fontsize = 22

avg_radius = np.zeros([N - N_start])
avg_overlap = np.zeros([N - N_start])


for i in range(N - N_start):

	avg_radius[i] = np.average( radii_data[:,i]  )
	avg_overlap[i] = np.average( overlap_data[:,i]  )


for r in range(N_reals):
	#Make plots of EcoSVM accuracy vs time
	plt.plot( radii_data[r,:] , linewidth=2, color = 'k')


plt.plot( avg_radius , linewidth=6, color = 'b')
plt.axhline(y = avg_batch_radius,linestyle='--',linewidth=6)
plt.ylabel("$ R(T) $",fontsize = fontsize + 2)
plt.xlabel("$ T  $" ,fontsize=fontsize + 2)
plt.grid()
plt.tick_params(labelsize=fontsize+2)
plt.tight_layout()
plt.show()
#plt.savefig("./graphs/SVDD_radii")
plt.clf()

for r in range(N_reals):
	#Make plots of EcoSVM number of support vectors vs time
	plt.plot(overlap_data[r,:], linewidth=2 , color = 'k')

plt.plot(avg_overlap, linewidth=6, color = 'b')
#plt.ylim(0.5,1.03)
plt.ylabel("$ S(T) $",fontsize = fontsize + 2)
plt.xlabel("$ T  $" ,fontsize=fontsize + 2)
plt.tick_params(labelsize=fontsize + 2)
plt.grid()
plt.tight_layout()
plt.show()
#plt.savefig("./graphs/SVDD_sim")
plt.clf()




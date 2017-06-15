from numpy import *
import random
from numpy.linalg import norm
from scipy import sparse
import time


############## Unit Objective function Function ##################3
def prob1(x,y,a,o):
	if o == 0: 								# Function value
		f = log(1 + exp(- y*dot(x,a)))
	elif o == 1:							# Function Gradient
		f = a * (-y* exp(- y*dot(x,a))/(1 + exp(- y*dot(x,a))))
	else:									# Hessian Matrix
		temp = y*y*(exp(- y*dot(x,a)))/( (1+exp(- y*dot(x,a)))*(1+exp(- y*dot(x,a)))) 
		a = sparse.csr_matrix(a)
		f = outer(a.T,a)* temp
	return f
	

############################### Stochastic gradient descent #####################
def sgd(prob1, x_0, y, a, N, step2):
	start = time.clock()
	nsample = len(a)
	M = 10								# memory parameter M
	L = 30								# number of iterations to compute correction parts again
	K = 1000							# number of maximal iterates
	alpha = step2						# steplength (need to set up later !!!!!!!!!!!!!!!!!)
	ntest = 50						# number of samples that are used to get the whole information
	t = -1 								# records number of correction pairs 
	w_t_cp = 0							# records correction pairs
	w_t_cp_new = w_t_cp
	k = 1								# records number of main iterations
	rand = random.randint(0, nsample-1) 	# find one random number between 1 and 20,242
	x_0 = ones([1,N])*0.1
	x = x_0						# set up initial point
	testset = random.sample(range(0,nsample), ntest)
	f = 0
	g = zeros([N,1]).T
	for item in testset:
		f = f + prob1(x,y[item],a[item,:],0)
		g = g + prob1(x,y[item],a[item,:],1).T
	f = f/len(testset)
	d = g
	g = g/len(testset)
	g_norm = norm(g)
	d_norm = norm(d)
	print ('======================== Iteration Begains =======================')
	print ('   k             f          ||g||       ||d||       alpha')
	print ('  {}       {}          {}          {}         {}'.format(k,f,g_norm,d_norm,alpha))
	while g_norm > 0.0001:
		w_t_cp_new = w_t_cp_new + x 						# accumulate for correction pairs
		if k <= 2*L:
			x_new  = x - alpha*g
		else:
			x_new = x - alpha*g
		x = x_new
		k = k + 1
		testset = random.sample(range(0,nsample), ntest)
		f = 0
		g = zeros([N,1]).T
		for item in testset:
			f = f + prob1(x,y[item],a[item,:],0)
			g = g + prob1(x,y[item],a[item,:],1)
		f = f/len(testset)
		d = g
		g = g/len(testset)
		g_norm = norm(g)
		d_norm = norm(d)
		print ('  {}        {}          {}          {}         {}'.format(k,f,g_norm,d_norm,alpha))
	end = time.clock() 
	timepass = start - end
	return x, timepass, k




############################ Two loop recursion ###############
def bfgshessian1(s_all,y_all,M,t,s,yh,x,q,N):
	start = time.clock()
	m_real = min(t,M)
	tempa = zeros([M,1]) 
	H_0 = sparse.eye(N)
	for j in range(t-1, t-m_real -1, -1):
		rho = 1/(s_all[j-t +m_real,:].dot(y_all[j-t+m_real,:].T))
		a1 = (rho*(s_all[j-t+m_real,:].dot(q.T)))
		tempa[j-(t-m_real)] = rho *(s_all[j-t+m_real,:].dot(q.T))
		q = q - a1*(y_all[j-t+m_real,:])
	H_0 = s_all[m_real - 1,:].dot(y_all[m_real-1,:].T)/y_all[m_real - 1,:].dot(y_all[m_real-1,:].T)
	r = H_0*q.T
	for j in range(t-m_real, t):
		rho = 1/(s_all[j-t+m_real,:].dot(y_all[j-t+m_real,:].T))
		if j != t-m_real:
			r = r.toarray()
		beta = rho*(y_all[j-t+m_real,:].dot(r))
		temp = s_all[j-t+m_real,:].T*((tempa[j-(t-m_real)]- beta))
		r = sparse.csr_matrix(r)
		temp = sparse.csr_matrix(temp)
		r =  r+ temp.T
	step = r.T
	return step


############################### Stochastic LBFGS #################3
def sgqn_test_hess(prob1, x_0, y, a, N, step1):
	start = time.clock()
	nsample = len(a)
	M = 10								# memory parameter M
	L = 30							# number of iterations to compute correction parts again
	K = 1000							# number of maximal iterates
	alpha = step1						# steplength (need to set up later !!!!!!!!!!!!!!!!!)
	ntest = 100					# number of samples that are used to get the whole information
	t = -1 								# records number of correction pairs 
	w_t_cp = zeros([1,N])						# records correction pairs
	w_t_cp_temp = w_t_cp
	w_t_cp_new = w_t_cp
	k = 1								# records number of main iterations
	s_all = zeros([M,N])				# store 10 s for lbfgs
	y_all = zeros([M,N])				# store 10 y for lbfgs
	x = x_0								# set up initial point
	testset = random.sample(range(0,nsample), ntest)
	f = 0
	g = zeros([1,N])
	testset = range(0,2)
	for item in testset:
		f = f + prob1(x,y[item],a[item,:],0)
		g = g + prob1(x,y[item],a[item,:],1)
	f = f/len(testset)
	d = g
	g = g/len(testset)
	g_norm = norm(g)
	d_norm = g_norm	
	print ('======================== Iteration Begains =======================')
	print ('   k        f         ||g||        alpha')
	print ('  {}     {}     {}       {}'.format(k,f,g_norm,alpha))
	while g_norm > 0.0001:
		w_t_cp_new = w_t_cp_new + x 						# accumulate for correction pairs
		if k <= 2*L:
		#if k <= 5000:
			x_new  = x - alpha*g
			norm(x_new)
			steplengthtemp = alpha
		else:
			q = g
			test1 = bfgshessian1(s_all,y_all,M,t,s,yh,x,q,N)
			x_new = x - alpha/((k-2*L))*test1
			x_new = array(x_new)
			steplengthtemp = alpha/(k-2*L)
		if k%L == 0:
			t = t + 1
			w_t_cp_new = w_t_cp_new/L
			if t > 0:
				testset = random.sample(range(0,nsample), 300)
				H_real = 0
				for item in testset:
					H_real = H_real + prob1(w_t_cp_new,y[item],a[item,:],2)[0,0]
				H = H_real/len(testset)
				s = w_t_cp_new - w_t_cp
				yh = s*H
				if t == 1:
					s_all[0,:] = s
					y_all[0,:] = yh
				elif (t <= M) and (t > 1):
					s_all[t-1,:] = s
					y_all[t-1,:] = yh
				else:
					s_all[range(0,M-1),:] = s_all[range(1,M),:]
					s_all[M-1,:] = s
					y_all[range(0,M-1),:] = y_all[range(1,M),:]
					y_all[M-1,:] = yh
			w_t_cp = w_t_cp_new
			w_t_cp_new = w_t_cp_temp
		x = x_new
		k = k + 1
		f = 0
		g = zeros([N,1]).T
		testset = random.sample(range(0,nsample), ntest)
		for item in testset:
			f = f + prob1(x,y[item],a[item,:],0)
			g = g + prob1(x,y[item],a[item,:],1)
		f = f/len(testset)
		d = g
		g = g/len(testset)
		g_norm = norm(g)
		d_norm = g_norm
		print ('  {}     {}     {}       {}'.format(k,f,g_norm,steplengthtemp))
	end = time.clock() 
	timepass = end - start
	return x_new, timepass, k


############################### Stochastic LBFGS with SVRG algorithm #################3
def sgqn_svrg(prob1, x_0, y, a, N, step1):
	start = time.clock()
	nsample = len(a)
	M = 10								# memory parameter M
	L = 30							# number of iterations to compute correction parts again
	K = 1000							# number of maximal iterates
	alpha = step1						# steplength (need to set up later !!!!!!!!!!!!!!!!!)
	MM  = 60
	ntest = 100					# number of samples that are used to get the whole information
	t = -1 								# records number of correction pairs 
	w_t_cp = zeros([1,N])						# records correction pairs
	w_t_cp_temp = w_t_cp
	w_t_cp_new = w_t_cp
	k = 1								# records number of main iterations
	s_all = zeros([M,N])				# store 10 s for lbfgs
	y_all = zeros([M,N])				# store 10 y for lbfgs
	x = x_0								# set up initial point
	v = x 								# for variable of SVRG
	testset = random.sample(range(0,nsample), ntest)
	f = 0
	g = zeros([1,N])
	testset = range(0,2)
	for item in testset:
		f = f + prob1(x,y[item],a[item,:],0)
		g = g + prob1(x,y[item],a[item,:],1)
	f = f/len(testset)
	d = g
	g = g/len(testset)
	g_norm = norm(g)
	d_norm = g_norm	
	print ('======================== Iteration Begains =======================')
	print ('   k        f         ||g||        alpha')
	print ('  {}     {}     {}       {}'.format(k,f,g_norm,alpha))	
	while g_norm > 0.0000001:
		w_t_cp_new = w_t_cp_new + x 						# accumulate for correction pairs
		if k <= 2*L:
		#if k <= 5000:
			x_new  = x - alpha*g
			norm(x_new)
			steplengthtemp = alpha
		else:
			q = g
			gsvrg = zeros([1,N])
			for item in testset:
				gsvrg = gsvrg + prob1(v,y[item],a[item,:],1)
			gsvrg = gsvrg/len(testset)
			q = q - gsvrg + svrg
			test1 = bfgshessian1(s_all,y_all,M,t,s,yh,x,q,N)
			x_new = x - alpha*test1
			x_new = array(x_new)
			steplengthtemp = alpha
		if k%L == 0:
			t = t + 1
			w_t_cp_new = w_t_cp_new/L
			if t > 0:
				testset = random.sample(range(0,nsample), 300)
				H_real = 0
				for item in testset:
					H_real = H_real + prob1(w_t_cp_new,y[item],a[item,:],2)[0,0]
				H = H_real/len(testset)
				s = w_t_cp_new - w_t_cp
				yh = s*H
				if t == 1:
					s_all[0,:] = s
					y_all[0,:] = yh
				elif (t <= M) and (t > 1):
					s_all[t-1,:] = s
					y_all[t-1,:] = yh
				else:
					s_all[range(0,M-1),:] = s_all[range(1,M),:]
					s_all[M-1,:] = s
					y_all[range(0,M-1),:] = y_all[range(1,M),:]
					y_all[M-1,:] = yh
			w_t_cp = w_t_cp_new
			w_t_cp_new = w_t_cp_temp
		x = x_new
		f = 0
		g = zeros([N,1]).T
		testset = random.sample(range(0,nsample), ntest)
		for item in testset:
			f = f + prob1(x,y[item],a[item,:],0)
			g = g + prob1(x,y[item],a[item,:],1)
		f = f/len(testset)
		d = g
		g = g/len(testset)
		g_norm = norm(g)
		d_norm = g_norm
		if k%MM == 1:
			lucktime = random.randint((k/MM)*MM, (k/MM + 1)*MM )
		if k == lucktime:
			vwait = x
		if k == 2*L:
			v = x
			svrg = zeros([1,N])
			for item in range(0,nsample):
				svrg = svrg + prob1(v,y[item],a[item,:],1)
			svrg = svrg/nsample
		if k%MM == 0:
			v = vwait
			svrg = zeros([1,N])
			for item in range(0,nsample):
				svrg = svrg + prob1(v,y[item],a[item,:],1)
			svrg = svrg/nsample
			x = v
		k = k + 1
		print ('  {}     {}     {}       {}'.format(k,f,g_norm,steplengthtemp))		
	end = time.clock() 
	timepass = end - start
	return x_new, timepass, k


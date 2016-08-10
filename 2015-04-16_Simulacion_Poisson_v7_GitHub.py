# -*- coding: utf-8 -*-
import numpy as np 
from numpy.random import poisson
import time
import os
import pickle
from scipy.stats import norm
import argparse

def mutual_pop_poisson(N1,N2,r,b,a,c,Dt):
	"""
	Created: 31-3-2016
	Modified: 31-3-2016
	"""
	return N1 + (np.sign(Dt*((r + b*N2)*N1 - (a + c*b*N2)*N1**2 )) *
		poisson(np.abs(Dt*((r + b*N2)*N1 - (a + c*b*N2)*N1**2 )  )))

def params_gauss(p1,p2,af,dg,eh,dt):
	"""
	Created: 31-3-2016
	Modified: 31-3-2016
	"""
	mu = dt*(af*p1 - dg*p1**2 - eh*p1*p2)
	sigma = np.abs(mu)
	return p1 + norm.rvs(mu,sigma)

def Mutualistic_Interaction_Poisson_Norm(
	R1_CI,R2_CI,B12_CI,B21_CI,
	N1_CI,N2_CI,
	r1_0,r2_0,b12_0,b21_0,
	r_RES,b_RES,
	alfa1,alfa2,c1,c2,
	f2_evol_type,
	f2steps=None,f2min=None,f2max=None,
	a1=0,d1=0,e1=0,h1=0,g1=0,f1=0,
	a2=1.0,d2=1.0,e2=1.0,g2=None,h2=None,
	Dt=0.001,
	dt=0.001,
	T=10000,
	v=False
	):
	print "Starting simulation Mutualistic_Interaction_Poisson_Norm"

	N1 = N1_CI
	N2 = N2_CI

	R1 = R1_CI
	R2 = R2_CI
	B12 = B12_CI
	B21 = B21_CI

	params_1 = []
	params_2 = []
	populations = []
	fix_points = []

	## Transicion entre espacios de fases
	F = generate_f_evolution(f2_evol_type,T,f2steps=f2steps,f2min=f2min,f2max=f2max)
	##############################################

	for i in xrange(T):
		if v:
			if i%10000 == 0:
				print "Step ",i
		## Asigno el valor de f2
		f2 = F[i]

		## Reescalo parametros
		r1 = (R1 + r1_0)*r_RES
		b12 = (B12  + b12_0)*b_RES
		r2 = (1.0*R2 + r2_0)*r_RES
		b21 = (10.0*B21 + b21_0)*b_RES

		############################## Calculo los puntos fijos
		A = complex(c2*b21*alfa1 + c1*b12*b21)
		B = complex(alfa1*alfa2 + c1*b12*r2 - c2*b21*r1 - b12*b21)
		C = complex(-r1*alfa2 - b12*r2)
		N1_fix1 = (-B + np.sqrt(B**2.0 - 4.0*A*C)) / (2.0*A)
		N1_fix2 = (-B - np.sqrt(B**2.0 - 4.0*A*C)) / (2.0*A)

		N2_fix1 = ( r2 + b21*N1_fix1 ) / (alfa2 + c2*b21*N1_fix1)
		N2_fix2 = ( r2 + b21*N1_fix2 ) / (alfa2 + c2*b21*N1_fix2)
		#############################

		## Guardo poblaciones, parametros y ptos fijos
		params_1.append((r1,b12))
		params_2.append((r2,b21))
		populations.append((N1,N2))
		fix_points.append([N1_fix1,N2_fix1,N1_fix2,N2_fix2])

		## Actualizo poblaciones
		N1_f = mutual_pop_poisson(N1,N2,r1,b12,alfa1,c1,Dt)
		N2_f = mutual_pop_poisson(N2,N1,r2,b21,alfa2,c2,Dt)

		## Actualizo parametros
		R1_f = 	params_gauss(R1,B12,a1,d1,e1,dt)
		B12_f = params_gauss(B12,R1,f1,g1,h1,dt)
		R2_f = 	params_gauss(R2,B21,a2,d2,e2,dt)
		B21_f = params_gauss(B21,R2,f2,g2,h2,dt)

		## Reasigno variables
		N1 = N1_f
		N2 = N2_f
		R1 = R1_f
		R2 = R2_f
		B12 = B12_f
		B21 = B21_f

	return populations, fix_points, params_1, params_2, F

def generate_f_evolution(f2_evol_type,T,f2steps=None,f2min=None,f2max=None):
	if f2_evol_type == 'continuos':
		F = np.linspace(f2min,f2max,T)
	elif f2_evol_type == 'discrete':
		if type(f2steps) == list:
			F_discrete = f2steps
		elif type(f2steps) == int:
			F_discrete = np.linspace(f2min,f2max,f2steps)
		else:
			raise Exception("Variable f2steps must be int or list")
		step_length = T / len(F_discrete)
		F = []
		for i in xrange(len(F_discrete)):
			F_val = F_discrete[i]
			F.extend([F_val]*step_length)
		## Para rellenar el hueco que pueda quedar al final del vector F
		F.extend([F_val]*(T-step_length*len(F_discrete)))
	elif f2_evol_type == 'constant':
		F = [f2min]*T
	else:
		raise Exception("Evolution type of parameter f unsupported")
	return F
	
if __name__ == '__main__':
	print "Starting program"

	parser = argparse.ArgumentParser()

	parser.add_argument("--save_results", default="./results",
					help="Change the save directory")

	parser.add_argument("-v", "--verbose", action="store_true", default=False,
					help="Increase output verbosity")

	parser.add_argument("-R1_CI", default=-1.0,type=float, help="Initial condition for r1. Def=-1.0")
	parser.add_argument("-B12_CI", default=6.0, type=float, help="Initial condition for b12. Def=6.0")
	parser.add_argument("-R2_CI", default=0.001, type=float, help="Initial condition for r2. Def=0.001")
	parser.add_argument("-B21_CI", default=0.8, type=float, help="Initial condition for b21. Def=0.8")

	parser.add_argument("-N1_CI", default=800, type=int, help="Initial condition for N1. Def=800")
	parser.add_argument("-N2_CI", default=800, type=int, help="Initial condition for N2. Def=800")
	
	parser.add_argument("-r1_0", default=0.0, type=float, help="Change of origin for r1. Def=0.0")
	parser.add_argument("-b12_0", default=0.0, type=float, help="Change of origin for b12. Def=0.0")
	parser.add_argument("-r2_0", default=-0.5, type=float, help="Change of origin for r2. Def=-0.5")
	parser.add_argument("-b21_0", default=-4.5, type=float, help="Change of origin for b21. Def=-4.5")

	parser.add_argument("-r_RES", default=1.0, type=float, help="Rescale factor for r. Def=1.0")
	parser.add_argument("-b_RES", default=1e-3, type=float, help="Rescale factor for b. Def=0.001")

	parser.add_argument("-alfa1", default=0.0001, type=float, help="Parameter alfa1 of the populations system. Def=0.0001")
	parser.add_argument("-alfa2", default=0.0001, type=float, help="Parameter alfa2 of the populations system. Def=0.0001")
	parser.add_argument("-c1", default=0.001, type=float, help="Parameter c1 of the populations system. Def=0.001")
	parser.add_argument("-c2", default=0.001, type=float, help="Parameter c2 of the populations system. Def=0.001")

	parser.add_argument("-a2", default=1.0, type=float, help="Parameter a of the environment system of the population 2. Def=1.0")
	parser.add_argument("-d2", default=1.0, type=float, help="Parameter d of the environment system of the population 2. Def=1.0")
	parser.add_argument("-e2", default=1.0, type=float, help="Parameter e of the environment system of the population 2. Def=1.0")
	parser.add_argument("-g2", default=2.0, type=float, help="Parameter g of the environment system of the population 2. Def=2.0")
	parser.add_argument("-h2", default=2.0/10.0, type=float, help="Parameter h of the environment system of the population 2. Def=0.2")

	parser.add_argument("-Dt", default=0.001, type=float, help="Time step of the population system. Def=0.001")
	parser.add_argument("-dt", default=0.01, type=float, help="Time step of the environment system. Def=0.01")
	parser.add_argument("-T", default=100000, type=int, help="Number of time steps of the simulation. Def=100000")

	parser.add_argument("-f2_evol_type", default='continuos',type=str, help="Type of evolution of parameter f2 (continuos,discrete or constant). Def=continuos")
	parser.add_argument("-f2steps",type=int, help="Number of steps of the discrete evolution of f2. Def=None")
	parser.add_argument("-f2min", default=-0.22,type=float, help="Starting value of f2. Def=-0.22")
	parser.add_argument("-f2max", default=2.2,type=float, help="Final value of f2. Def=2.2")

	args = parser.parse_args()

	save_results = args.save_results + "/"

	if not os.path.isdir(save_results):
		os.mkdir(save_results)

	R1_CI = args.R1_CI
	B12_CI = args.B12_CI
	R2_CI = args.R2_CI
	B21_CI = args.B21_CI
	
	N1_CI = args.N1_CI
	N2_CI = args.N2_CI
	
	r1_0 = args.r1_0
	b12_0 = args.b12_0
	r2_0 = args.r2_0
	b21_0 = args.b21_0
	r_RES = args.r_RES
	b_RES = args.b_RES

	alfa1 = args.alfa1
	alfa2 = args.alfa2
	c1 = args.c1
	c2 = args.c2

	a2 = args.a2
	d2 = args.d2
	e2 = args.e2
	g2 = args.g2
	h2 = args.h2

	Dt = args.Dt
	dt = args.dt
	T = args.T

	f2_evol_type = args.f2_evol_type 
	f2steps = args.f2steps
	f2min = args.f2min
	f2max = args.f2max

	v = args.verbose

	t0 = time.time()
	populations, fix_points, params_1, params_2, F = Mutualistic_Interaction_Poisson_Norm(
		R1_CI,R2_CI,B12_CI,B21_CI,
		N1_CI,N2_CI,
		r1_0,r2_0,b12_0,b21_0,
		r_RES,b_RES,
		alfa1,alfa2,c1,c2,
		f2_evol_type=f2_evol_type,
		f2steps=f2steps,f2min=f2min,f2max=f2max,
		a1=0,d1=0,e1=0,h1=0,g1=0,f1=0,
		a2=a2,d2=d2,e2=e2,g2=g2,h2=h2,
		Dt=Dt,
		dt=dt,
		T=T,
		v=v)
	print "Total time ", (time.time()-t0)/60.0, "min"

	print "Savint data"
	
	with open(save_results + "parameters.txt","w") as f:
		f.write(
			"R1_CI = %f\n\
B12_CI = %f\n\
R2_CI = %f\n\
B21_CI = %f\n\
N1_CI = %f\n\
N2_CI = %f\n\
r1_0 = %f\n\
b12_0 = %f\n\
r2_0 = %f\n\
b21_0 = %f\n\
r_RES = %f\n\
b_RES = %f\n\
alfa1 = %f\n\
alfa2 = %f\n\
c1 = %f\n\
c2 = %f\n\
g2 = %f\n\
h2 = %f\n\
Dt = %f\n\
dt = %f\n\
T = %f"%(R1_CI,B12_CI,R2_CI,B21_CI,N1_CI,N2_CI,r1_0,b12_0,r2_0,b21_0,r_RES,b_RES,alfa1,alfa2,c1,c2,g2,h2,Dt,dt,T)
			)
	N1, N2 = zip(*populations)
	N1_fix1,N2_fix1,N1_fix2,N2_fix2 = zip(*fix_points)
	r1,b12 = zip(*params_1)
	r2,b21 = zip(*params_2)

	data_dict_out = {"N1":N1,
	"N2":N2,
	"N1_fix1":N1_fix1,
	"N1_fix2":N1_fix2,
	"N2_fix1":N2_fix1,
	"N2_fix2":N2_fix2,
	"r1":r1,
	"r2":r2,
	"b12":b12,
	"b21":b21,
	"f2":F}

	for k,v in data_dict_out.iteritems():
		with open(save_results + "%s.txt"%k,"wb") as f:
			for i in v:
				f.write(str(i)+"\n")
	print "Data saved"
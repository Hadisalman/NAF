#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:33:04 2018

@author: chinwei
"""

import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
#import torch.optim as optim
from torchkit import optim
from torch.autograd import Variable
from torchkit import nn as nn_, flows, utils
from torchkit.transforms import from_numpy, binarize
from torchvision.transforms import transforms
from ops import load_bmnist_image, load_omniglot_image, load_mnist_image
from ops import DatasetWrapper
from itertools import chain

import time
import json
import argparse, os

from IPython import embed
from mag.experiment import Experiment

def logdensity_1(z):
	z1, z2 = torch.split(z, [1,1], 1) 
	norm = torch.sqrt(z1 ** 2 + z2 ** 2)
	exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
	exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
	u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
	return -u

class VAE(object):
	
	def __init__(self, args):

		self.args = args        
		self.__dict__.update(args.__dict__)
		
		dimz = args.dimz
		dimc = args.dimc
		dimh = args.dimh
		flowtype = args.flowtype
		num_flow_layers = args.num_flow_layers
		num_ds_dim = args.num_ds_dim
		num_ds_layers = args.num_ds_layers
		
		self.dimh = dimh
				 
		act = nn.ELU()
		if flowtype == 'affine':
			flow = flows.IAF
		elif flowtype == 'dsf':
			flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
												 num_ds_layers=num_ds_layers,
												 **kwargs)
		elif flowtype == 'ddsf':
			flow = lambda **kwargs:flows.IAF_DDSF(num_ds_dim=num_ds_dim,
												  num_ds_layers=num_ds_layers,
												  **kwargs)
		
		self.enc = nn.Sequential(
					torch.nn.Linear(2, dimc),
					torch.nn.ReLU(),
					torch.nn.Linear(dimc, dimc),
					torch.nn.ReLU(),

				)

		
		self.inf = nn.Sequential(
				# flows.LinearFlow(dimz, dimc),
				*[nn_.SequentialFlow(
					flow(dim=dimz,
						 hid_dim=dimh,
						 context_dim=dimc,
						 num_layers=2,
						 activation=act),
					flows.FlipFlow(1)) for i in range(num_flow_layers)])

		if self.cuda:
			self.enc = self.enc.cuda()
			self.inf = self.inf.cuda()

		
		amsgrad = bool(args.amsgrad)
		polyak = args.polyak
		self.optim = optim.Adam(chain(self.enc.parameters(),
									  self.inf.parameters()),
								lr=args.lr, 
								betas=(args.beta1, args.beta2),
								amsgrad=amsgrad,
								polyak=polyak)
		
		
	
	def loss(self, n, weight=1.0, bits=0.0):
		# n = x.size(0)
		zero = utils.varify(np.zeros(1).astype('float32'))
		
		# z_samples = np.random.normal(0.0, 1.0, [config.batch_size,2])       
		ep = utils.varify(np.random.randn(n,self.dimz).astype('float32'))
		context = self.enc(ep)

		ep = utils.varify(np.random.randn(n,self.dimz).astype('float32'))	
		lgd = utils.varify(np.zeros(n).astype('float32'))
		if self.cuda:
			ep = ep.cuda()
			lgd = lgd.cuda()
			zero = zero.cuda()

		z, logdet, _ = self.inf((ep, lgd, context))
		# pi = nn_.sigmoid(self.dec(z))
		
		# logpx = - utils.bceloss(pi, x).sum(1).sum(1).sum(1)
		logqz = utils.log_normal(ep, zero, zero).sum(1) - logdet
		# logpz = utils.log_normal(z, zero, zero).sum(1)
		logpz = logdensity_1(z).sum(1)
		kl = logqz - logpz
		
		return kl, kl
		
	
	def clip_grad_norm(self):
		nn.utils.clip_grad_norm(chain(self.inf.parameters()),
								self.clip)



class model(object):
	
	def __init__(self, args, filename):
		
		self.__dict__.update(args.__dict__)

		self.filename = filename
		self.args = args        
		
		if args.final_mode:
			tr = np.concatenate([tr, va], axis=0)
			va = te[:]
		
		self.vae = VAE(args)
		
		
	def train(self, epoch, total=10000):
		optim = self.vae.optim
		t = 0 
		
		LOSSES = 0
		KLS = 0
		counter = 0
		best_val = float('inf')
		n=100
		for e in range(epoch):            
			optim.zero_grad()
			weight = min(1.0,max(self.anneal0, t/float(total)))
			losses_, kl = self.vae.loss(n, weight, self.bits)
			losses_.mean().backward()
			losses = kl
			LOSSES += losses.sum().data.cpu().numpy()
			KLS += kl.sum().data.cpu().numpy()
			counter += losses.size(0)

			self.vae.clip_grad_norm()
			optim.step()
			t += 1

			if e%300 == 0:
				print("Loss on iteration {}: {}".format(e , LOSSES/float(counter)))
				zero = utils.varify(np.zeros(1).astype('float32'))
				ep = utils.varify(np.random.randn(10000,self.dimz).astype('float32'))
				context = self.vae.enc(ep)
				lgd = utils.varify(np.zeros(10000).astype('float32'))
				if self.cuda:
					ep = ep.cuda()
					lgd = lgd.cuda()
					zero = zero.cuda()

				ep = utils.varify(np.random.randn(10000,self.dimz).astype('float32'))	

				output, _, _ = self.vae.inf((ep, lgd, context))
				# embed()
				plot_density_from_samples(			
					samples=output.data.numpy(),
					directory=experiment.reconstructed_distribution,
					iteration=e,
					flow_length=args.num_flow_layers,
					)

			LOSSES = 0
			KLS = 0
			counter = 0

		

# =============================================================================
# main
# =============================================================================

def plot_density_from_samples(samples, directory, iteration, flow_length, X_LIMS=(-7,7), Y_LIMS=(-7,7)):
	import matplotlib.pyplot as plt
	import numpy as np
	from scipy.stats import kde

	# create data
	x = samples[:,0]
	y = samples[:,1]

	# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
	nbins=300
	k = kde.gaussian_kde([x,y], bw_method=.1)
	x1 = np.linspace(*X_LIMS, 300)
	x2 = np.linspace(*Y_LIMS, 300)
	xi, yi = np.meshgrid(x1, x2)
	zi = k(np.vstack([xi.flatten(), yi.flatten()]))

	# Make the plot
	fig = plt.figure(figsize=(7, 7))
	plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='hot')
	plt.colorbar()
	plt.title(
		"Flow length: {}\n Samples on iteration #{}"
		.format(flow_length, iteration)
	)
	plt.axis('equal')
	# plt.axis('tight')
	fig.savefig(os.path.join(directory, "flow_result_{}.png".format(iteration)))
	plt.close()

"""parsing and configuration"""
def parse_args():
	desc = "VAE"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset', type=str, default='sb_mnist', 
						choices=['sb_mnist', 
								 'db_mnist',
								 'db_omniglot'],
						help='static/dynamic binarized mnist')
	parser.add_argument('--epoch', type=int, default=20000, 
						help='The number of epochs to run')
	parser.add_argument('--batch_size', type=int, default=100, 
						help='The size of batch')
	parser.add_argument('--save_dir', type=str, default='models',
						help='Directory name to save the model')
	parser.add_argument('--result_dir', type=str, default='results',
						help='Directory name to save the generated images')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='Directory name to save training logs')
	parser.add_argument('--seed', type=int, default=1993,
						help='Random seed')
	parser.add_argument('--fn', type=str, default='0',
						help='Filename of model to be loaded')
	parser.add_argument('--to_train', type=int, default=1,
						help='1 if to train 0 if not')
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--clip', type=float, default=5.0)
	parser.add_argument('--beta1', type=float, default=0.9)
	parser.add_argument('--beta2', type=float, default=0.999)
	parser.add_argument('--anneal', type=int, default=50000)
	parser.add_argument('--anneal0', type=float, default=0.0001)
	parser.add_argument('--bits', type=float, default=0.10)
	parser.add_argument('--amsgrad', type=int, default=0)
	parser.add_argument('--polyak', type=float, default=0.998)
	parser.add_argument('--cuda', default=False, action='store_true')
	parser.add_argument('--final_mode', default=False, action='store_true')
	

	parser.add_argument('--dimz', type=int, default=2)
	parser.add_argument('--dimc', type=int, default=2)
	parser.add_argument('--dimh', type=int, default=2)
	parser.add_argument('--flowtype', type=str, default='dsf')
	parser.add_argument('--num_flow_layers', type=int, default=2)
	parser.add_argument('--num_ds_dim', type=int, default=2)
	parser.add_argument('--num_ds_layers', type=int, default=1)
	
	parser.add_argument('--exp-name', type=str, default='temp',
					help='experiment name')
	
	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
	# --save_dir
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	# --result_dir
	if not os.path.exists(args.result_dir + '_' + args.dataset):
		os.makedirs(args.result_dir + '_' + args.dataset)

	# --result_dir
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	# --epoch
	try:
		assert args.epoch >= 1
	except:
		print('number of epochs must be larger than or equal to one')

	# --batch_size
	try:
		assert args.batch_size >= 1
	except:
		print('batch size must be larger than or equal to one')

	return args
   

"""main"""
def main():

	np.random.seed(args.seed)
	torch.manual_seed(args.seed+10000)

	fn = str(time.time()).replace('.','')
	print(args)
	print(fn)

	print(" [*] Building model!")    
	old_fn = args.save_dir+'/'+args.fn+'_args.txt'
	if os.path.isfile(old_fn):
		def without_keys(d, keys):
			return {x: d[x] for x in d if x not in keys}
		d = without_keys(json.loads(open(old_fn,'r').read()),
						 ['to_train','epoch','anneal'])
		args.__dict__.update(d)
		print(" New args:" )
		print(args)
		mdl = model(args, fn)
		print(" [*] Loading model!")
		mdl.load(args.save_dir+'/'+args.fn)
	else:
		mdl = model(args, fn)
	
	# launch the graph in a session
	if args.to_train:
		print(" [*] Training started!")
		mdl.train(args.epoch, args.anneal)
		print(" [*] Training finished!")



# parse arguments
args = parse_args()
if args is None:
	exit()

experiment = Experiment({"exp_name": args.exp_name},
					experiments_dir='./experiments'
					)

config = experiment.config
experiment.register_directory("distributions")
experiment.register_directory("postTrainingAnalysis")
experiment.register_directory("samples")
experiment.register_directory("reconstructed_distribution")
experiment.register_directory("invertedSamples")

if __name__ == '__main__':
	main()

	
	

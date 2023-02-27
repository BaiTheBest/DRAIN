# -*- coding: utf-8 -*-

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise


def dataset_preparation(args, num_tasks=10, num_instance=220):

    if args.dataset in ['Elec2','HousePrice','M5Hobby','M5Household','Energy']:
        # A = np.load('data/{}/processed/A.npy'.format(args.dataset))
        # U = np.load('data/{}/processed/U.npy'.format(args.dataset))
        X = np.load('data/{}/X.npy'.format(args.dataset))
        Y = np.load('data/{}/Y.npy'.format(args.dataset))
    else:
        # A = np.load('data/{}/processed/A.npy'.format(args.dataset))
        # U = np.load('data/{}/processed/U.npy'.format(args.dataset))
        X = np.load('data/{}/processed/X.npy'.format(args.dataset))
        Y = np.load('data/{}/processed/Y.npy'.format(args.dataset))
    
    dataloaders = []

    if args.dataset == 'Moons':
        intervals = np.arange(num_tasks+1)*num_instance
    elif args.dataset == 'ONP':
        intervals = np.array([0,7049,13001,18725,25081,32415,39644])
    elif args.dataset == 'Elec2':
        intervals = np.array([0,670,1342,2014,2686,3357,4029,4701,5373,6045,6717,7389,8061,8733,
            9405,10077,10749,11421,12093,12765,13437,14109,14781,15453,16125,16797,17469,18141,18813,
            19485,20157,20829,21501,22173,22845,23517,24189,24861,25533,26205,26877,27549])
    elif args.dataset == 'HousePrice':
        intervals = np.array([0,2119,4982,8630,12538,17079,20937,22322])
    elif args.dataset == 'M5Hobby':
        intervals = np.array([0,323390,323390*2,323390*3,997636])
    elif args.dataset == 'M5Household':
        intervals = np.array([0,124100,124100*2,124100*3,382840])
    elif args.dataset == 'Energy':
        intervals = np.array([0,2058,2058+2160,2058+2*2160,2058+3*2160,2058+4*2160,2058+5*2160,2058+6*2160,2058+7*2160,19735])

    for i in range(len(intervals)-1):
        temp_X = X[intervals[i]:intervals[i+1]]
        temp_Y = Y[intervals[i]:intervals[i+1]]
        domain_dataset = DomainDataset(temp_X,temp_Y) # create dataset for each domain
        temp_dataloader = DataLoader(domain_dataset, batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers)
        dataloaders.append(temp_dataloader)
    
    return dataloaders

class DomainDataset(Dataset):
    """ Customized dataset for each domain"""
    def __init__(self,X,Y):
        self.X = X                           # set data
        self.Y = Y                           # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]    # return list of batch data [data, labels]
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse

# Import model
from model import RNN
# Import functions
from utils import dataset_preparation, make_noise

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
if not os.path.isdir('logs'):
    os.makedirs('logs')
log_file = 'logs/log_{}.log'.format(datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str): logger.info(str)


log('Is GPU available? {}'.format(torch.cuda.is_available()))
#print('Is GPU available? {}'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2']
parser.add_argument("--dataset", default="Moons", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))

# Hyper-parameters
parser.add_argument("--noise_dim", default=16, type=float,
                    help="the dimension of the LSTM input noise.")
parser.add_argument("--num_rnn_layer", default=1, type=float,
                    help="the number of RNN hierarchical layers.")
parser.add_argument("--latent_dim", default=16, type=float,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--hidden_dim", default=8, type=float,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--noise_type", choices=["Gaussian", "Uniform"], default="Gaussian",
                    help="The noise type to feed into the generator.")

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")
parser.add_argument("--epoches", default=20, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--batch_size", default=16, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="the unified learning rate for each single task.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

args = parser.parse_args()

def train(dataloader, optimizer, rnn_unit, args, task_id=0, input_E=None, input_hidden=None):
    E = input_E
    hidden = input_hidden
    log("Start Training on Domain {}...".format(task_id))
    for epoch in range(args.epoches):
        accs = []
        with tqdm(dataloader, unit="batch") as tepoch:
            for X, Y in tepoch:
                tepoch.set_description("Task_ID: {} Epoch {}".format(task_id, epoch))
                
                X, Y  = X.float().to(device), Y.float().to(device)
                initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
                
                #  Training on Single Domain
                rnn_unit.train()
                optimizer.zero_grad()
                E, hidden, pred = rnn_unit(X, initial_noise, E, hidden)
                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                loss = F.binary_cross_entropy(pred.squeeze(-1), Y)
                
                prediction = torch.as_tensor((pred.detach() - 0.5) > 0).float()
                accuracy = (prediction.squeeze(-1) == Y).float().sum()/prediction.shape[0]
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
                accs.append(accuracy.item())
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item())
            
            
            #log("Task_ID: {}\tEpoch: {}\tAverage Training Accuracy: {}".format(task_id, epoch, np.mean(accs)))
    return E, hidden, rnn_unit
    

def evaluation(dataloader, rnn_unit, args, input_E, input_hidden):
    rnn_unit.eval()
    E = input_E
    hidden = input_hidden
    test_accs = []
    log("Start Testing...")
    with tqdm(dataloader, unit="batch") as tepoch:
        for X, Y in tepoch:                
            X, Y  = X.float().to(device), Y.float().to(device)
            initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
            with torch.no_grad():
                _, _, pred = rnn_unit(X, initial_noise, E, hidden)
                loss = F.binary_cross_entropy(pred.squeeze(-1), Y)
                
                prediction = torch.as_tensor((pred.detach() - 0.5) > 0).float()
                accuracy = (prediction.squeeze(-1) == Y).float().sum()/prediction.shape[0]  
                test_accs.append(accuracy.item())
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    log("Average Testing Accuracy is {}".format(np.mean(test_accs)))


def main(arsgs):
    output_directory='outputs-{}'.format(args.dataset)
    model_directory='models-{}'.format(args.dataset)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)
        
    log('use {} data'.format(args.dataset))
    log('-'*40)
    
    if args.dataset == 'Moons':
        num_tasks=10
        data_size=2
        num_instances=220
    elif args.dataset == 'MNIST':
        num_tasks=11
        data_size=2
        num_instances=200
    elif args.dataset == 'ONP':
        num_tasks=6
        data_size=58
        num_instances=None
    elif args.dataset == 'Elec2':
        num_tasks=41
        data_size=8
        num_instances=None
    # Defining dataloaders for each domain
    dataloaders = dataset_preparation(args, num_tasks, num_instances)
    rnn_unit = RNN(data_size, device, args).to(device)
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(rnn_unit.parameters(), lr=args.learning_rate)

    starting_time = time.time()
    
    # Training
    Es, hiddens = [None], [None]
    for task_id, dataloader in enumerate(dataloaders[:-1]):
        E, hidden, rnn_unit = train(dataloader, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1])
        Es.append(E)
        hiddens.append(hidden)
        print("========== Finished Task #{} ==========".format(task_id))

    ending_time = time.time()

    print("Training time:", ending_time - starting_time)
    
    # Testing
    evaluation(dataloaders[-1], rnn_unit, args, Es[-1], hiddens[-1])
        

if __name__ == "__main__":
    print("Start Training...")
    
    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    main(args)










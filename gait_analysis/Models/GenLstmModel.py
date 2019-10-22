import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler
import numpy as np
import time

from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os, gc
import psutil
from sys import getsizeof

from gait_analysis.utils.files import set_logger
from gait_analysis.utils import training

from gait_analysis import CasiaDataset
from gait_analysis.Config import Config
from gait_analysis import Composer
# from guppy import hpy
from gait_analysis.utils import files
# GLOBAL VARIABLES
c = Config()

time_stamp = training.get_time_stamp()
logger, log_folder =  set_logger('FLOWS40',c,time_stamp=time_stamp, level='INFO')

def get_dataloaders(dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(c.config['network']['validation_split'] * dataset_size))
    if c.config['network']['shuffle_dataset']:
        np.random.seed(c.config['network']['randomized_seed'])
        np.random.shuffle(indices)
    train_indices , test_indices = indices[split:] , indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset , batch_size=c.config['network']['BATCH_SIZE'] , sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset , batch_size=c.config['network']['BATCH_SIZE'] , sampler=test_sampler)
    return train_loader, test_loader

def get_optimizer(model,learning_rate=None,momentum=None):
    '''
    Creates an optimizer.
    If no input parameters takes configuration defined in the Config object.
    :param learning_rate:
    :param momentum:
    :return:
    '''
    if not learning_rate:
        learning_rate = c.config['network']['learning_rate']
    if not momentum:
        momentum = c.config['network']['momentum']

    optimizer = optim.SGD(model.parameters() , lr= learning_rate,
                          momentum=momentum)
    return optimizer
def get_dataset():
    composer = Composer()
    transformer = composer.compose()
    return CasiaDataset(transform=transformer)

def test(model,dataloader,device='cpu'):
    logger.info('Start testing...')
    correct = 0
    total = 0
    with torch.no_grad():
        for i , batch in enumerate(dataloader):
            inputs , labels = batch
            scenes = [s.to(device) for s in inputs['flows']]
            labels = labels.to(device)
            if not labels.size()[0] == c.config['network']['BATCH_SIZE']:
                # skip uncompleted batch size NN is fixed to BATCHSIZE
                continue
            outputs = model(scenes)
            _ , predicted = torch.max(outputs.data , 1)
            n_errors = torch.nonzero(torch.abs(labels.long() - predicted)).size(0)
            total += predicted.numel()
            correct += predicted.numel() - n_errors
    if total==0:
        logger.info('Warning: no enough data to perform the test')
        return
    logger.info('Accuracy {:.2f}%'.format(100 * correct / total))
    logger.info('...testing finished')

# @profile
def train(model,optimizer, criterion, train_loader,test_loader=None, device='cpu'):
    if not test_loader:
        test_loader = train_loader

    n_batches = len(train_loader)
    logger.info('number of batches in the train loader: {}'.format(n_batches))
    # Time for printing
    training_start_time = time.time()
    learning_rate = c.config['network']['learning_rate']
    logger.info('Start training...')
    train_loss_hist = np.zeros(c.config['network']['epochs'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, threshold=1e-7)
    inputs_dev, labels_dev = training.get_training_vectors_device(train_loader , 'flows' , device)

    pid = os.getpid()
    logger.info('Process running in PID: {}'.format(pid))
    py = psutil.Process(pid)
    prev_mem = 0
    it = 0
    # torch.backends.cudnn.enabled = False

    for epoch in range(c.config['network']['epochs']):
        logger.info("Epoch: {}/{}".format(epoch+1,c.config['network']['epochs']))
        print_every = n_batches // 10
        if print_every == 0:
            print_every = 1
        start_time = time.time()
        total_train_loss = 0
        running_loss = 0.0
        total_train_loss = train_epoch(criterion , epoch , inputs_dev , it , labels_dev , model , n_batches ,
                                       optimizer , prev_mem , print_every , py , running_loss , start_time ,
                                       total_train_loss , train_loader)
        logger.info('total training loss for epoch {}: {:.6f}'.format(epoch + 1 , total_train_loss))
        train_loss_hist[epoch] = total_train_loss
        scheduler.step(total_train_loss)

    logger.info('...Training finished. Total time of training: {:.2f} [mins]'.format((time.time()-training_start_time)/60))
    plot_file_name = "{0}/{1}-{2}".format(log_folder , time_stamp , c.config['logger']['plot_file'])
    training.plot_train_loss_hist(train_loss_hist , save=True , filename=plot_file_name)
    logger.info('saving figure in: {}'.format(plot_file_name))
    return model

# @profile()
def run_batch(inputs , labels , optimizer , model , criterion,py):
    logger.info(" ====> MEMORY AT THE BEGINNING OF THE BATCH: {}".format(py.memory_info()[0] / 2. ** 20 ))
    optimizer.zero_grad()
    outputs = model(inputs)
    logger.debug("====> Raw Out: {} {}".format(len(outputs) , outputs.size()))
    logger.debug("====> Raw Labels: {} {}".format(len(labels) , labels.size()))
    logger.debug("====> Out: {} {}".format(len(outputs) , outputs.size()))
    logger.debug("====> Labels: {} {}".format(len(labels) , labels.size()))
    loss = criterion(outputs , labels)
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().numpy()





# @profile()
def train_epoch(criterion , epoch , inputs_dev , it , labels_dev , model , n_batches , optimizer , prev_mem ,
                print_every , py , running_loss , start_time , total_train_loss , train_loader):
    for i , batch in enumerate(train_loader):
        inputs , labels = batch
        if not labels.size()[0] == c.config['network']['BATCH_SIZE']:
            # skip uncompleted batch size NN is fixed to BATCH_SIZE
            continue
        for ii , s in enumerate(inputs['flows']):
            inputs_dev[ii].copy_(s)
        labels_dev.copy_(labels)
        loss = run_batch(inputs_dev, labels_dev, optimizer, model, criterion, py)
        logger.info(" ===> MEMORY AT THE END OF THE BATCH: {}".format(py.memory_info()[0] / 2. ** 20))
        # Print statistics
        running_loss += loss #loss.detach().data.item()
        total_train_loss += loss #loss.detach().data.item()
        # Print every 10th batch of an epoch
        if (i + 1) % (print_every + 1) == 0:
            logger.info("Epoch {}, {:d}% \t train_loss(mean): {:.2f} took: {:.2f}s".format(
                epoch + 1 , int(100 * (i + 1) / n_batches) , running_loss / print_every , time.time() - start_time))
            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
        gc.collect()
        # memory look at:
        logger.info('---------------------------------------------------------')
        it += 1
        cur_mem = py.memory_info()[0] / 2. ** 20  # memory use in MB...I think
        add_mem = cur_mem - prev_mem
        prev_mem = cur_mem
        logger.info("train iterations: {}, added mem: {}M, current mem: {}M".format(it , add_mem , cur_mem))

    return total_train_loss


def main():
    # TRAINING
    # Defines the device (cuda:0) is available
    device = torch.device(c.config['network']['device'] if torch.cuda.is_available() else "cpu")
    logger.info("Device in usage: {}".format(device))

    # creates the network
    model = CNNLSTM()
    model.to(device)

    # Defines new criterion
    criterion = nn.CrossEntropyLoss()

    # creates optimizer
    optimizer = get_optimizer(model)

    # instantiates dataset
    dataset = get_dataset()
    logger.info('dataset lenght: {}'.format(len(dataset)))
    logger.info('dataset elements: {}'.format(dataset.dataset_items))

    # creates dataloders
    train_dataloader, test_dataloader = get_dataloaders(dataset)

    # training
    logger.info('configuration: {}'.format(c.config))
    model = train(model,optimizer,criterion,train_dataloader,device=device)

    # testing
    logger.info('Testing in the training set:...')
    test(model , train_dataloader , device)
    logger.info('Testing in the testing set:...')
    test(model , test_dataloader , device)

    # save model
    model_file_name = "{0}/{1}-{2}".format(log_folder , time_stamp , c.config['logger']['model_file'])
    training.save_model(model_file_name)

    plt.show()

if __name__== '__main__':
    main()
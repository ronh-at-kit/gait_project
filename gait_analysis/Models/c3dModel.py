import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler
import numpy as np
import time

from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os, gc
import psutil

from gait_analysis.utils.files import set_logger
from gait_analysis.utils import training

from gait_analysis import CasiaDataset
from gait_analysis.Config import Config
from gait_analysis import Composer
from gait_analysis.utils import files
import gait_analysis.Models.resnet as resnet

# GLOBAL VARIABLES
c = Config()
time_stamp = training.get_time_stamp()
logger, log_folder =  set_logger('pretrained3d',c,time_stamp=time_stamp, level='INFO')
#logger, log_folder =  set_logger('test_delete_after',c,time_stamp=time_stamp, level='DEBUG')

class CNN3DPretrained(nn.Module):
    def __init__(self):
        super(CNN3DPretrained, self).__init__()
        self.c = Config()
        self.avialable_device = torch.device(c.config['network']['device'] if torch.cuda.is_available() else "cpu")
        model = resnet.resnet101(
                num_classes=400,
                shortcut_type='B',
                sample_size=8,
                sample_duration=20)
        pretrained_dict = torch.load(c.config['network']['pretrain_path'])
        self.resnet = model.load_state_dict(pretrained_dict['state_dict'])
        self.flat_feat = 256
        self.fc1s = []
        self.fc2s = []
        for i in range(c.config['network']['TIMESTEPS']):
            self.fc1s.append(nn.Linear(self.flat_feat, 128))
            self.fc2s.append(nn.Linear(128 , 3))
            self.fc1s[i].to(device=self.avialable_device)
            self.fc2s[i].to(device=self.avialable_device)


    def forward(self , x):
        #         print("Input list len:",len(x))
        #         print("Input elemens size:", x[0].size())
        #         c.config['network']['BATCH_SIZE'] = x[0].size()[0]
        x = torch.stack(x).transpose(0,1).transpose_(1,2)
        with torch.no_grad:
            x = self.resnet(x)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0),-1)
        output = []
        for i in range(c.config['network']['TIMESTEPS']):
            z = x
            z = F.relu(self.fc1s[i](z))
            z = F.relu(self.fc2s[i](z))
            output.append(z)
        output = torch.stack(output,1).transpose(1,2)
        return output


class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.c = Config()
        self.avialable_device = torch.device(c.config['network']['device'] if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv3d(3, 16, (1, 5, 5), stride=(1,2,2))  # input 640x480
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)  # input 638x478 output 319x239
        self.conv2 = nn.Conv3d(16, 32, (1, 5, 5), stride=(1,2,2))  # input 640x480
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)  # input 638x478 output 319x239
        self.conv3 = nn.Conv3d(32, 32, (1, 5, 5), stride=(1, 2, 2))  # input 640x480
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2)  # input 638x478 output 319x239

        # self.conv3 = nn.Conv3d(16, c.config['network']['TIMESTEPS'], (5, 5, 5), stride=(1,2,2))  # input 640x480
        # self.bn3 = nn.BatchNorm3d( c.config['network']['TIMESTEPS'])
        # self.pool3 = nn.MaxPool3d(2, 2)  # input 638x478 output 319x239
        self.flat_feat = 256
        self.fc1s = []
        self.fc2s = []
        for i in range(c.config['network']['TIMESTEPS']):
            self.fc1s.append(nn.Linear(self.flat_feat, 128))
            self.fc2s.append(nn.Linear(128 , 3))
            self.fc1s[i].to(device=self.avialable_device)
            self.fc2s[i].to(device=self.avialable_device)


    def forward(self , x):
        #         print("Input list len:",len(x))
        #         print("Input elemens size:", x[0].size())
        #         c.config['network']['BATCH_SIZE'] = x[0].size()[0]
        x = torch.stack(x).transpose(0,1).transpose_(1,2)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0),-1)
        output = []
        for i in range(c.config['network']['TIMESTEPS']):
            z = x
            z = F.relu(self.fc1s[i](z))
            z = F.relu(self.fc2s[i](z))
            output.append(z)
        output = torch.stack(output,1).transpose(1,2)
        return output


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
    else:
        train_indices , test_indices = indices[split:] , indices[:split]
        train_sampler = SequentialSampler(train_indices)
        test_sampler = SequentialSampler(test_indices)
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
    logger.info('Create optimizer SGD with lr = {} and momentum = {}'.format(learning_rate,momentum))
    optimizer = optim.SGD(model.parameters() , lr= learning_rate,
                          momentum=momentum)
    optimizer.param_groups
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
            scenes = [s.to(device) for s in inputs['scenes']]
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


def pred_errors(labels, outputs):
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        n_errors = torch.nonzero(torch.abs(labels.long() - predicted)).size(0)
        total = predicted.numel()
    return n_errors, total

def train(model , optimizer , criterion , train_loader , scheduler=None , epoch_count=0, loss=0, hist = [], device=torch.device('cpu')):

    ## INITIALIZATIONS:
    # print every calculation
    n_batches = len(train_loader)
    logger.info('number of batches in the train loader: {}'.format(n_batches))
    print_every = n_batches // 10
    if print_every == 0:
        print_every = 1

    # Time for printing
    training_start_time = time.time()

    # initialize train loss history and epoch_counter
    epoch_total = c.config['network']['epochs'] + epoch_count
    if len(hist)>0 and len(hist.shape) == 1:
        # back-compatibility for hist vector in 1d
        hist = hist.reshape((1,len(hist)))
        hist = np.concatenate((hist,np.zeros_like(hist)),axis=0)
        logger.warning('Backcomptibility bug will be generated in the plots')
    if len(hist)>0:
        train_hist = np.concatenate((hist,np.zeros((2,c.config['network']['epochs']))),axis=1)
    else:
        train_hist = np.zeros((2, c.config['network']['epochs']))


    # initialize vectors train
    inputs_dev, labels_dev = training.get_training_vectors_device(train_loader , 'scenes' , device)
    ## TRAINING
    logger.info("======== Start Training ==========")
    for epoch in range(epoch_count,epoch_total):
        logger.info("Epoch: {}/{}".format(epoch+1,epoch_total))
        start_time = time.time()
        epoch_loss, epoch_accuracy = train_epoch(criterion, epoch, inputs_dev, labels_dev, model, n_batches, optimizer,
                                       print_every, start_time, train_loader)
        logger.info('total training loss for epoch {}: {:.6f}'.format(epoch + 1 , epoch_loss))
        logger.info('total training accuracy for epoch {}: {:.6f}'.format(epoch + 1 , epoch_accuracy))
        train_hist[0][epoch] = epoch_loss
        train_hist[1][epoch] = epoch_accuracy
        if scheduler:
            scheduler.step(np.round(epoch_loss,decimals=2))
    ## SAVE AND RETURN COMPUTATIONS
    logger.info('...Training finished. Total time of training: {:.2f} [mins]'.format((time.time()-training_start_time)/60))
    return model, epoch_loss, train_hist




def run_batch(inputs , labels , optimizer , model , criterion):
    optimizer.zero_grad()
    outputs = model(inputs)
    logger.debug("====> Raw Out: {} {}".format(len(outputs) , outputs.size()))
    logger.debug("====> Raw Labels: {} {}".format(len(labels) , labels.size()))
    logger.debug("====> Out: {} {}".format(len(outputs) , outputs.size()))
    logger.debug("====> Labels: {} {}".format(len(labels) , labels.size()))
    loss = criterion(outputs , labels.long())
    loss.backward()
    optimizer.step()
    n_errors, total = pred_errors(labels, outputs)
    accuracy = 100*(total-n_errors)/total
    logger.debug("====> Accuracy of the batch {}".format(accuracy))
    return loss.detach().cpu().numpy(), accuracy, n_errors, total

def train_epoch(criterion, epoch, inputs_dev, labels_dev, model, n_batches, optimizer, print_every, start_time,
                train_loader):
    epoch_loss = 0.0
    batch_loss = 0.0
    batch_accuracy = 0.0
    epoch_errors = 0
    epoch_samples = 0
    for i , batch in enumerate(train_loader):
        inputs , labels = batch
        if not labels.size()[0] == c.config['network']['BATCH_SIZE']:
            # skip uncompleted batch size NN is fixed to BATCH_SIZE
            continue
        for ii , s in enumerate(inputs['scenes']):
            inputs_dev[ii].copy_(s)
        labels_dev.copy_(labels)
        loss, accuracy, n_errors, total = run_batch(inputs_dev, labels_dev, optimizer, model, criterion)
        # Print statistics
        epoch_loss += loss
        batch_loss += loss
        epoch_errors += n_errors
        epoch_samples += total
        batch_accuracy += accuracy
        # Print every 10th batch of an epoch
        if (i + 1) % (print_every) == 0:
            logger.info("Epoch {}, {:d}% stats (avg./batch): loss={:.2f} accuracy={}% time={:.2f}s".format(
                epoch + 1 , int(100 * (i + 1) / n_batches) , batch_loss / print_every ,batch_accuracy / print_every , time.time() - start_time))
            # Reset running loss and time
            batch_loss = 0.0
            batch_accuracy =0.0
            start_time = time.time()
    epoch_accuracy = 100*(epoch_samples - epoch_errors)/epoch_samples
    return epoch_loss, epoch_accuracy


def main(input_path=None,lr=None):
    # TRAINING
    # Defines the device (cuda:0) is available
    target_device = c.config['network']['device'] if torch.cuda.is_available() else "cpu"
    device = torch.device(target_device)
    logger.info("Device in usage: {}".format(device))

    # creates the network
    # model = CNN3D()
    model = CNN3DPretrained()
    if input_path:
        model = training.load_model(input_path, model, target=target_device)
        model.eval()
        model.train()
    else:
        model.to(device)

    # Defines new criterion
    criterion = nn.CrossEntropyLoss()

    # creates optimizer
    optimizer = get_optimizer(model, learning_rate=lr)
    if input_path:
        optimizer = training.load_optimizer(input_path,optimizer,target=target_device)

    # instantiates dataset
    dataset = get_dataset()
    logger.info('dataset lenght: {}'.format(len(dataset)))
    logger.info('dataset elements: {}'.format(dataset.dataset_items))

    # creates dataloders
    train_dataloader, test_dataloader = get_dataloaders(dataset)

    # creates scheduler
    # scheduler = None
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, threshold=1e-7)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
    # training
    logger.info('configuration: {}'.format(c.config))
    if input_path:
        loss, hist, epoch = training.load_status(input_path)
        model , loss , hist = train(model , optimizer , criterion , train_dataloader , scheduler=scheduler ,
                                    loss=loss, hist=hist, epoch_count=epoch, device=device)
    else:
        epoch = 0
        model, loss, hist = train(model , optimizer , criterion , train_dataloader , scheduler=scheduler , device=device)

    # plot hist
    plt.figure()
    plot_file_name_loss = "{0}/{1}-loss-{2}".format(log_folder , time_stamp , c.config['logger']['plot_file'])
    training.plot_train_loss_hist(hist[0] , save=True , filename=plot_file_name_loss)
    plt.figure()
    plot_file_name_accu = "{0}/{1}-accu-{2}".format(log_folder, time_stamp, c.config['logger']['plot_file'])
    training.plot_train_accu_hist(hist[1], save=True, filename=plot_file_name_accu)
    logger.info('saving figure in: {}'.format(plot_file_name_loss))
    logger.info('saving figure in: {}'.format(plot_file_name_accu))

    # testing
    logger.info('Testing in the training set:...')
    test(model , train_dataloader , device)
    logger.info('Testing in the testing set:...')
    test(model , test_dataloader , device)

    # save model
    model_file_name = "{0}/{1}-{2}".format(log_folder , time_stamp , c.config['logger']['model_file'])
    logger.info('saving model at: {}'.format(model_file_name))
    training.save_model(model_file_name , model , optimizer=optimizer , epoch=epoch+c.config['network']['epochs'] , loss=loss , hist=hist ,
                        source=target_device)

    plt.show()

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Flow-lstm script')
    parser.add_argument('-m', '--model', metavar='path to model', type=str,
                        help='path to the model')
    parser.add_argument('-l', '--lr', metavar='learning rate', help='value of the training rate', type=float)
    args = parser.parse_args()
    input_path = files.correct_path(args.model) if args.model else None
    learning_rate = args.lr if args.lr else None
    main(input_path, learning_rate)
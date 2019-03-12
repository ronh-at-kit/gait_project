import torch
from gait_analysis.Config import Config
import matplotlib
import datetime
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys

def save_model(path , model , optimizer=None , epoch=0 , loss=0 , hist=[] , source='cpu'):
    '''
    you can save a dict to be load as:
    {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    :param path:
    :return:
    '''
    if optimizer:
        torch.save({
        'epoch': epoch ,
        'model_state_dict': model.state_dict() ,
        'optimizer_state_dict': optimizer.state_dict() ,
        'loss': loss,
        'hist': hist,
        'source': source},
            path)
    else:
        torch.save({
        'epoch': epoch ,
        'model_state_dict': model.state_dict() ,
        'optimizer_state_dict': None,
        'loss': loss,
        'hist': hist,
        'source': source},
            path)


def load_model(path, model, target=None):
    '''
    load a model in a form a dictionary and initialize the model and the optimizer
    :param source: it Is the device from where we are reading.
    :param target: it is the device in where we load the model.
    :param path: path where the model is stored
    :param model: model to be parametrized
    :param optimizer: optimizer to be parametrized/
    :return: model, optiizer,epoch and loss
    '''
    checkpoint = torch.load(path)
    source = checkpoint['source']
    c = Config()
    device = torch.device(target)
    if source=='cpu' and target=='cpu':
        # CPU to CPU
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif not source == 'cpu' and target == 'cpu':
        # GPU to CPU
        checkpoint = torch.load(path , map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif not source == 'cpu' and not target == 'cpu':
        # GPU to GPU
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    else:
        # CPU to GPU
        checkpoint = torch.load(path , map_location=c.config['network']['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    return model

def load_optimizer(path, optimizer, target):
    checkpoint = torch.load(path)
    if checkpoint['optimizer_state_dict']:
        source = checkpoint['source']
        c = Config()
        device = torch.device(target)
        if source == 'cpu' and target == 'cpu':
            # CPU to CPU
            checkpoint = torch.load(path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif not source == 'cpu' and target == 'cpu':
            # GPU to CPU
            checkpoint = torch.load(path , map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif not source == 'cpu' and not target == 'cpu':
            # GPU to GPU
            checkpoint = torch.load(path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #optimizer.to(device)
        else:
            # CPU to GPU
            checkpoint = torch.load(path , map_location=c.config['network']['device'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #optimizer.to(device)
        return optimizer
    else:
        return optimizer

def load_status(path):
    checkpoint = torch.load(path)
    return checkpoint['loss'], checkpoint['hist'], checkpoint['epoch']

def plot_train_loss_hist(train_loss_hist, save=False, filename=None):
    c = Config()
    plt.clf()
    plt.plot(train_loss_hist[train_loss_hist!=0])
    plt.title('train loss history')
    plt.xlabel('epoch number')
    plt.ylabel('train loss for all epoch')
    plt.draw()
    if save:
        plt.savefig(filename)
def plot_train_accu_hist(train_accu_hist, save=False, filename=None):
    c = Config()
    plt.clf()
    plt.plot(train_accu_hist[train_accu_hist != 0])
    plt.title('train accuracy history')
    plt.xlabel('epoch number')
    plt.ylabel('train accuracy for all epochs')
    plt.draw()
    if save:
        plt.savefig(filename)



def get_time_stamp():
    return str(datetime.datetime.now()).replace(' ' , '_').replace(':' , 'i')

def get_training_vectors_device(dataloader , field , device):

    inputs , labels = iter(dataloader).next()
    input_init = torch.zeros_like(inputs[field][0])
    labels_init = torch.zeros_like(labels)
    inputs_init = [input_init.to(device) for s in inputs[field]]
    labels_init = labels_init.to(device)
    return inputs_init, labels_init

def print_memory_size(environment, logger=None):
    if logger:
        for var , obj in environment.items():
            if not var.startswith('__'):
                logger.info("memory size ({}): {}  ".format(var , get_size(obj)))
        logger.info('---------------------------------------------------------')
    else:
        for var , obj in environment.items():
            if not var.startswith('__'):
                print("memory size ({}): {}  ".format(var , get_size(obj)))
        print('---------------------------------------------------------')


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
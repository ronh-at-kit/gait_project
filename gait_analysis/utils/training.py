import torch
from gait_analysis.Config import Config
import matplotlib
import datetime
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys

def save_model(path,model, optimizer = None, epoch=0, loss=0):
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
        'loss': loss},
            path)
    else:
        torch.save({
        'epoch': epoch ,
        'model_state_dict': model.state_dict() ,
        'optimizer_state_dict': None,
        'loss': loss},
            path)


def load_model(path, model, optimizer, source=None, target=None):
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
    print('checkpoint[optimizer_state_dict]',checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    if checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def plot_train_loss_hist(train_loss_hist, save=False, filename=None):
    c = Config()
    plt.clf()
    plt.plot(train_loss_hist[train_loss_hist!=0])
    plt.title('train loss history')
    plt.xlabel('epoch number')
    plt.ylabel('train loss for all epoch')
    plt.draw()
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
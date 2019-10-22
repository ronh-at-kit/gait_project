import numpy as np
import torch

class WeightWatcher(object):
    """Monitors weight changes in layers
    Initialize class with list of layers to monitor and call function after e.g. each training epoch
    Only
    """
    # TODO LOSS
    def __init__(self,net,layers):
        # layer_list = ['conv1', 'fc3']
        print("Layers:",net.parameters)
        print("Input: e.g. ['conv1','features[3]','classifier[2]'")
        self.layer_init_dic = {}
        self.layer_var = {}
        self.layer_mean = {}
        for layer in layers:
            weights = self.__get_model_weights(net,layer)
            self.layer_init_dic[layer] = weights
            self.layer_var[layer] = []
            self.layer_mean[layer] = []

    def update_weights(self,net,layers):
        for layer in layers:
            weights_tmp = self.__get_model_weights(net,layer)
            weights_init = self.layer_init_dic[layer]
            mean = np.mean(weights_tmp - weights_init)
            var = np.var(weights_tmp - weights_init)
            self.layer_mean[layer].append(mean)
            self.layer_var[layer].append(var)

    def __get_model_weights(self,net,str):
        # this is nasty with "exec(), but one line"
        exec("self._weights = list(net." + str + ".parameters())[0].detach().numpy().flatten()")
        return self._weights

    def get_weight_changes(self,layers):
        if len(layers) == 1:
            return self.layer_var[layers[0]]
        else:
            var_list = []
            for layer in layers:
                var_list.append(self.layer_var[layer])
            return var_list


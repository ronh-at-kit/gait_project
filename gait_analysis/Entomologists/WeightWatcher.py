import numpy as np

class WeightWatcher(object):
    """Monitors weight changes in layers
    Initialize class with list of layers to monitor and call function after e.g. each training epoch
    Only
    """
    # TODO LOSS
    def __init__(self,net,layers):
        # layer_list = ['conv1', 'fc3']
        self.layer_init_dic = {}
        self.layer_var = {}
        self.layer_mean = {}
        for layer in layers:
            weights = self.__get_model_weights(net,layer)
            self.layer_init_dic[layer] = weights
            self.layer_var[layer] = []
            self.layer_mean[layer] = []

    # def __call__(self,model):
    #     print("Do nothing")
    #     # model = test_net

    def update_weights(self,net,layers):
        for layer in layers:
            weights_tmp = self.__get_model_weights(net,layer)
            weights_init = self.layer_init_dic[layer]
            mean = np.mean(weights_tmp - weights_init)
            var = np.var(weights_tmp - weights_init)
            self.layer_mean[layer].append(mean)
            self.layer_var[layer].append(var)
            # print("Mean",mean)
            # print("Var",var)

    def __get_model_weights(self,net,str):
        # this is nasty with "exec()"
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

        # if str == 'conv1':
        #     return list(net.conv1.parameters())[0].detach.numpy().flatten()
        # elif str == 'conv2':
        #     return list(net.conv2.parameters())[0].detach.numpy().flatten()
        # elif str == 'conv3':
        #     return list(net.conv3.parameters())[0].detach.numpy().flatten()
        # elif str == 'conv4':
        #     return list(net.conv4.parameters())[0].detach.numpy().flatten()
        # elif str == 'conv5':
        #     return list(net.conv5.parameters())[0].detach.numpy().flatten()
        # elif str == 'fc1':
        #     return list(net.fc1.parameters())[0].detach.numpy().flatten()
        # elif str == 'fc2':
        #     return list(net.fc2.parameters())[0].detach.numpy().flatten()
        # elif str == 'fc3':
        #     return list(net.fc3.parameters())[0].detach.numpy().flatten()



import csv
import torch

class AccuracyTracker(object):
    """Takes labels and predictions and analyzes prediciton accuracy on each class
    Assumes that the labels are [0,1,2] since our class will always have those labels
    TODO: confuison matrix possible etc.
    """
    def __init__(self,possible_labels):
        self.different_labels = possible_labels
        self.correctly = [0] * len(self.different_labels)
        self.total = [0] * len(self.different_labels)
        self.acc_graph = []
        for _ in range(len(possible_labels)):
            self.acc_graph.append([])
        self.total_acc_graph = []
        self.loss_graph = []
        self.lr_graph = []

    def reset_acc(self):
        self.correctly = [0]*len(self.different_labels)
        self.total = [0]*len(self.different_labels)

    def update_acc(self,predictions,labels):
        # input: torch list, suitable for update with purely convolutional net, update maybe later
        for prediction, label in zip(predictions, labels):
            for true_label in self.different_labels:
                if true_label == label and label != prediction:
                    self.total[true_label] += 1
                    # confusion matrix update
                elif prediction == label and label == true_label:
                    self.correctly[true_label] += 1
                    self.total[true_label] += 1

    def update_graph(self):
        for i in range(len(self.different_labels)):
            self.acc_graph[i].append(self.get_acc()[i])
        self.total_acc_graph.append(self.get_total_acc())

    def get_acc_graph(self):
        return self.acc_graph

    def get_acc_tot_graph(self):
        return self.total_acc_graph

    def reset_graph(self):
        for i in range(len(possible_labels)):
            self.acc_graph[i] = []
        self.total_acc_graph = []

    def update_loss(self,loss):
        if type(loss) == torch.Tensor:
            self.loss_graph.append(loss.data.item())
        elif type(loss) == float:
            self.loss_graph.append(loss)
        else:
            print("Warning, unknown type, only accepts torch.Tensor or float (from either loss or loss.data.item()")

    def reset_loss(self):
        self.loss_graph = []

    def get_loss_graph(self):
        return self.loss_graph

    def update_lr(self,lr):
        self.lr_graph.append(lr)

    def reset_lr(self):
        self.lr_graph = []

    def get_lr_graph(self):
        return self.lr_graph

    # def reset_graph(self):
    # def get_acc_graph(self)

    def get_acc(self):
        total_mod = [1 if word == 0 else word for word in self.total]
        # returns 0 if no elements in this class
        acc = [c/t for c,t in zip(self.correctly,total_mod)]
        return acc

    def get_labels_distribution(self):
        return self.total

    def get_abs_predictions(self):
        return self.correctly

    def get_total_acc(self):
        total_mod = [1 if word == 0 else word for word in self.total]
        return sum(self.correctly)/sum(total_mod)

    def write_to_csv(self,path):
        with open(path + "acc_tot.csv", 'w') as file:
            wr = csv.writer(file)
            wr.writerow(self.total_acc_graph)
        with open(path + "loss.csv", 'w') as file:
            wr = csv.writer(file)
            wr.writerow(self.loss_graph)
        with open(path + "lr.csv", 'w') as file:
            wr = csv.writer(file)
            wr.writerow(self.lr_graph())
        for i in range(len(self.acc_graph)):
            with open(path + "acc" + str(i) + ".csv", 'w') as file:
                wr = csv.writer(file)
                wr.writerow(self.acc_graph[i])
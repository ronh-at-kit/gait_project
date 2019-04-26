from gait_analysis.Entomologists import AccuracyTracker

class AccuracyTrackerTrainTest(object):
    """AccuracyTracker for both Training and Test set
    """
    def __init__(self,possible_labels):
        self.different_labels = possible_labels
        self.train_tracker = AccuracyTracker.AccuracyTracker(possible_labels)
        self.test_tracker = AccuracyTracker.AccuracyTracker(possible_labels)

    def update_loss(self,loss):
        self.train_tracker.update_loss(loss)
        self.test_tracker.update_loss(loss)

    def update_lr(self,lr):
        self.train_tracker.update_lr(lr)
        self.test_tracker.update_lr(lr)

    def write_to_csv(self, path):
        self.train_tracker.write_to_csv(path + 'train/')
        self.test_tracker.write_to_csv(path + 'test/')

    def update_acc(self, predictions, labels,str):
        if str == "TEST":
            self.test_tracker.update_acc(predictions,labels)
        elif str == "TRAIN" or str == "TRAINING":
            self.train_tracker.update_acc(predictions,labels)
        else:
            print("Acc not updated. Please specify")

    def update_graph(self):
        self.train_tracker.update_graph()
        self.test_tracker.update_graph()

    def reset_acc_both(self):
        self.train_tracker.reset_acc()
        self.test_tracker.reset_acc()

    def get_acc(self,str):
        if str == "TRAIN" or str == "TRAINING":
            return self.train_tracker.get_acc()
        elif str == "TEST":
            return self.test_tracker.get_acc()
        else:
            print("Acc not specified. Return Training")
            return self.train_tracker.get_acc()

    def get_acc_tot(self,str):
        if str == "TRAIN" or str == "TRAINING":
            acc =  self.train_tracker.get_acc_tot()
        elif str == "TEST":
            acc = self.test_tracker.get_acc_tot()
        else:
            print("Acc not specified. Return Training")
            acc = self.train_tracker.get_acc_tot()
        return acc

    def get_labels_distribution(self,str):
        if str == "TRAIN" or str == "TRAINING":
            return self.train_tracker.total
        elif str == "TEST":
            return self.test_tracker.total
        else:
            print("Distribution not specified. Return Training")
            return self.train_tracker.total



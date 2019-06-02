#TODO define abstract Data Loader

class DataLoader:
    def iterate_train(self, shuffle=True):
        pass

    def iterate_validation(self, shuffle=True):
        pass




class SimpleDataLoader(DataLoader):
    keypoint_files = None
    def __init__(self, keypoint_files):
        pass






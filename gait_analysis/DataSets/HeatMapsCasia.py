import cv2
import os
from os.path import join
from os import listdir


from torch.utils.data import Dataset
from gait_analysis.Config import Config
import gait_analysis.settings as settings

class HeatMapsCasia(Dataset):
    def __init__(self, dataset_items, transform=None):
        self.data_items = dataset_items
        self.heatmap_dir = settings.casia_heatmaps_dir
        self.config = Config()
        self.options = {}


    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        dataset_item = self.data_items[idx]
        grouping = self.config.get_indexing_grouping()
        if grouping == 'person_sequence':
            p_num , sequence = dataset_item
            # by default we select the 90 degree grom person_sequence grouping
            angle = 90
        elif grouping == 'person_sequence_angle':
            p_num , sequence , angle = dataset_item
        return self.__load_headmap(p_num, sequence,angle)

    def set_option(self,key, value):
        self.options[key] = value

    def __load_headmap(self,p_num,sequence,angle):
        scene_folder = os.path.join(self.heatmap_dir , '{:03d}'.format(p_num) , sequence \
                                    , '{}-{:03d}'.format(sequence , angle) )
        # TODO: this code is repetive for images put in loading module
        def compose_image_filename(angle , i):
            return join(scene_folder,'{:03d}-{}-{:03d}_frame_{:03d}_pose_heatmaps.png'.format(p_num , sequence , angle , i))
        head_map = []
        if 'valid_indices' in self.options:
            valid_indices = self.options['valid_indices']
            head_map += [compose_image_filename(angle , i) for i , valid in enumerate(valid_indices) if valid]
        else:
            head_map += [join(scene_folder , f) for f in listdir(scene_folder)
                            if f.endswith('.png')]
            head_map.sort()

        def read_image(im_file):
            if not os.path.isfile(im_file):
                raise ValueError('{} don\'t exist.'.format(im_file))
            im = cv2.imread(im_file , cv2.IMREAD_UNCHANGED)
            # im = cv2.cvtColor(im , cv2.COLOR_BGR2RGB)
            return im

        def separate(head_map_concatenated):
            width = 496
            # height = 368
            return [head_map_concatenated[:,width*i:(i+1)*width] for i in range(0,25)]

        scene_images = [separate(read_image(f)) for f in head_map]
        return scene_images
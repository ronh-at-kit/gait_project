import os
import gait_analysis.settings as settings
from gait_analysis.utils.data_loading import list_annotations_files


class CasiaItemizer():
    def __init__(self,mode):
        self.mode = mode

    def get_items(self):
        annotation_files = list_annotations_files(settings.casia_annotations_dir)
        dataset_items = list(map(self.__extract_pnum_subsequence, annotation_files))
        return dataset_items

    def __extract_pnum_subsequence(self,path):
        filename = path.split(os.path.sep)[-1]
        p_num = filename[0:3]
        subsequence = filename[4:9]
        return (int(p_num), subsequence)
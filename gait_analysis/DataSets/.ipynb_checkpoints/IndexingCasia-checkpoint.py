import os
import  itertools
import gait_analysis.settings as settings
from gait_analysis.utils.data_loading import list_annotations_files
from gait_analysis.Config import Config

class IndexingCasia():
    def __init__(self):
        self.config = Config()

    def get_items(self):

        selection = self.config.get_indexing_selection()
        # 1. Selection
        if selection == 'auto':
            # selection of all squencei quta habve final annotaion
            annotation_files = list_annotations_files(settings.casia_annotations_dir)
        elif selection == 'manual_people':
            # selection of people in the list that have final annotaions
            people_selection = self.config.config['indexing']['people_selection']
            all_annotation_files = list_annotations_files(settings.casia_annotations_dir)
            annotation_files = [ f for f in all_annotation_files if int(f[-27:-24]) in people_selection]
        elif selection == 'manual_people_sequence':
            people_selection = self.config.config['indexing']['people_selection']
            sequences_selection = self.config.config['indexing']['sequences_selection']
            people_sequences_selection = [ '{:03d}-{}'.format(a,b) \
                                           for a,b in list(itertools.product(people_selection, sequences_selection))]
            all_annotation_files = list_annotations_files(settings.casia_annotations_dir)

            annotation_files = [f for f in all_annotation_files if f[-27:-18] in people_sequences_selection]
        else:
            annotation_files = list_annotations_files(settings.casia_annotations_dir)


        # 2. Grouping...
        grouping = self.config.get_indexing_grouping()

        if grouping == 'person_sequence':
            dataset_items = list(map(self.__extract_person_sequence , annotation_files))
        if grouping == 'person_sequence_angle':
            PSA_generators  = list(map(self.__extract_person_sequence_angle , annotation_files))
            dataset_items = [item for generator in PSA_generators for item in generator]

        return dataset_items

    def __extract_person_sequence(self , path):
        filename = path.split(os.path.sep)[-1]
        p_num = filename[0:3]
        sequence = filename[4:9]
        return (int(p_num), sequence)

    def __extract_person_sequence_angle(self , path):
        angles = [18,54,90,126,162]
        filename = path.split(os.path.sep)[-1]
        p_num = filename[0:3]
        sequence = filename[4:9]
        for a in angles:
            yield (int(p_num),sequence,a)

if __name__ == '__main__':
    c = Config()
    config = c.config
    ## default configuration
    print('default:')
    indexing = IndexingCasia()
    items = indexing.get_items()
    print(items)
    ## by selection people
    print('selection manual_people:')
    config['indexing']['selection'] = 'manual_people'
    config['indexing']['people_selection'] = list(range(1,4))
    items = indexing.get_items()
    print(items)
    ## by selection people sequce
    config['indexing']['selection'] = 'manual_people_sequence'
    config['indexing']['people_selection'] = [1,40]
    config['indexing']['sequences_selection'] = ['bg-01','cl-01']
    print('selection manual people sequence')
    items = indexing.get_items()
    print(items)

    ## by selection people sequce and grouping person sequnece angle
    config['indexing']['selection'] = 'manual_people_sequence'
    config['indexing']['people_selection'] = [1,3,4,5,6]+list(range(50,56))
    config['indexing']['sequences_selection'] = ['bg-01' , 'cl-01']
    config['indexing']['grouping'] = 'person_sequence_angle'
    print('selection people sequce and grouping person sequnece angle')
    items = indexing.get_items()
    print(items)
print("Settings.py should be e.g. crops")
from gait_analysis import CasiaDataset
from gait_analysis.Config import Config
import numpy as np

from gait_analysis import CasiaDataset
from gait_analysis.Config import Config
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

c = Config()
c.config['indexing']['grouping'] = 'person_sequence_angle'

MARGIN = 10
CROP_SIZE = [156.0, 230.0]
CALC_CROP_SIZE = False
CROP_FLOWS = True
if not CALC_CROP_SIZE:
    IMAGE_OUTPUT_SIZE = [CROP_SIZE[0]+2*MARGIN,CROP_SIZE[1]+2*MARGIN]
    print("image output size",IMAGE_OUTPUT_SIZE)
else:
    print("image output size to be determined with [CROP_SIZE[0]+2*MARGIN,CROP_SIZE[1]+2*MARGIN]")
PADDING = False

dataset = CasiaDataset()

# print("Using", poses.shape[0],"poses")
invalid_pose_counter = 0
last_person = '000'
print("Person56 has a problem with flows")

for item, i in zip(dataset, range(len(dataset))):
    annotations = item['annotations']

    if CROP_FLOWS:
        data_in = item['flows']
    else:
        data_in = item['scenes']

    poses = item['poses']

    # since valid scenes have an offset respect to thier annotation number
    annotations_offset = int(annotations[''][0])

    person = '{:03d}'.format(dataset.dataset_items[i][0])
    sequence = dataset.dataset_items[i][1]
    angle = '{:03d}'.format(dataset.dataset_items[i][2])
    origin = '/home/ron/PycharmProjects/Gait2019/CASIA/'
    folderpath_for_debug_only = origin + 'images/' + person + '/' + sequence + '/' + angle + '/'
    if CROP_FLOWS:
        pathlib.Path(origin + 'preprocessing/crops_flow/' + person + '/' + sequence + '/' + sequence + '-' + angle + '/'
                     ).mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(origin + 'preprocessing/crops/' + person + '/' + sequence + '/' + sequence + '-' + angle + '/'
                     ).mkdir(parents=True, exist_ok=True)
    #     print("current folder: ", folderpath)

    if last_person != person:
        print("Person", person)
    last_person = person

    x0_last = 1
    x1_last = 1
    y0_last = 1 + CROP_SIZE[0]
    y1_last = 1 + CROP_SIZE[1]
    if len(data_in) != len(poses):
        print("Warning, unequal length of flows and poses:", len(data_in), len(poses))
    for j in range(min(len(data_in), len(poses))):
        pose_x = poses[j][:, 0]
        pose_y = poses[j][:, 1]
        if (all(x == 0 for x in pose_x) or all(y == 0 for y in pose_y)):
            print("WARNING: Invalid poses in ", folderpath_for_debug_only, "pose nr", j,
                  "taking last available pose (or crop size if first)")
            x0 = x0_last
            x1 = x1_last
            y0 = y0_last
            y1 = y1_last
            invalid_pose_counter += 1
        else:
            # HERE x is horizontal direction and y is vertical direction
            x0 = np.floor(np.min(pose_x)) - MARGIN
            x1 = np.floor(np.max(pose_x)) + MARGIN
            y0 = np.floor(np.min(pose_y)) - MARGIN
            y1 = np.floor(np.max(pose_y)) + MARGIN

        # make sure image borders stay in range
        x0 = max(0, x0)
        x1 = min(x1, data_in[j].shape[1])
        y0 = max(0, y0)
        y1 = min(y1, data_in[j].shape[0])

        #       padding image until it gets the desired size
        x_to_pad = IMAGE_OUTPUT_SIZE[0] - (x1 - x0)
        y_to_pad = IMAGE_OUTPUT_SIZE[1] - (y1 - y0)
        x_pad_l = np.floor(x_to_pad / 2)
        x_pad_r = x_to_pad - x_pad_l
        y_pad_l = np.floor(y_to_pad / 2)
        y_pad_r = y_to_pad - y_pad_l

        if PADDING:
            im_tmp = data_in[j][int(y0):int(y1), int(x0):int(x1)]
            im_final = np.pad(im_tmp, [(int(y_pad_l), int(y_pad_r)), (int(x_pad_l), int(x_pad_r)), (0, 0)],
                              'constant')  # ,'constant', constant_values=((0, 0),(0,0)))
        else:  # NOT PADDING IMAGE WITH ZEROS
            x0 = x0 - x_pad_l
            x1 = x1 + x_pad_r
            y0 = y0 - y_pad_l
            y1 = y1 + y_pad_r
            #             print("Coordinates before",x0_n,x1_n,y0_n,y1_n)
            if x0 < 0:
                x1 = x1 - x0
                x0 = 0
            elif data_in[j].shape[1] < x1:
                x0 = x0 - x1 + data_in[j].shape[1]
                x1 = data_in[j].shape[1]
            if y0 < 0:
                y1 = y1 - y0
                y1 = 0
            elif data_in[j].shape[0] < y1:
                y0 = y0 - y1 + data_in[j].shape[0]
                y1 = data_in[j].shape[0]

            im_final = data_in[j][int(y0):int(y1), int(x0):int(x1)]

            if (x1 - x0 != IMAGE_OUTPUT_SIZE[0] or y1 - y0 != IMAGE_OUTPUT_SIZE[1]):
                print("WRONG IMAGE COORDINATES DETECTED ", folderpath_for_debug_only, "pose nr", j)
                print("Coordinates after", x0, x1, y0, y1)
        plt.imshow(im_final)
        framename = '{:03d}'.format(j + annotations_offset)
        if CROP_FLOWS:
            saving_path = origin + 'preprocessing/crops_flow/' + person + '/' + sequence + '/' + sequence + '-' + angle + '/'
        else:
            saving_path = origin + 'preprocessing/crops/' + person + '/' + sequence + '/' + sequence + '-' + angle + '/'
        filename = person + '-' + sequence + '-' + angle + '_frame_' + framename + '_flow.png'
        totalpath = saving_path + filename
        #         print("Image name", filename)
        #         print("Total path", totalpath)
        Image.fromarray(im_final).save(totalpath)
        x0_last = x0
        x1_last = x1
        y0_last = y0
        y1_last = y1

print("Done")
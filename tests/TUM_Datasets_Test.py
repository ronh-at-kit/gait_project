import sys

sys.path.insert(0, '..')
from gait_analysis import TumGAID_Dataset
from gait_analysis import settings
from gait_analysis.utils.openpose_utils import maybe_subtract_poses


def test_tumgaid():
    tumgaid_default_args = settings.tumgaid_default_args
    tumgait_dataset = TumGAID_Dataset(settings.tumgaid_root,
                                      settings.tumgaid_preprocessing_root,
                                      settings.tumgaid_annotations_root,
                                      tumgaid_default_args)

    output, annotations = tumgait_dataset[1]
    for k in output['flow_maps']:
        for kk in k:
            assert (kk.shape == (10, 10, 2)), '{}'.format(kk.shape)

    assert len(output['pose_keypoints']) == len(annotations)
    # flow maps are always pairwise, therfore -1
    assert len(output['flow_maps']) == (len(annotations) - 1)


# TODO test case that pose it outside image


# TODO add CI


def test_openpose_utils():
    tumgaid_args = settings.tumgaid_default_args
    tg_dset = TumGAID_Dataset(settings.tumgaid_root,
                              settings.tumgaid_preprocessing_root,
                              settings.tumgaid_annotations_root,
                              tumgaid_args)
    data, annotations = tg_dset[0]
    pose_keypoints = data['pose_keypoints']
    diff_keypoints = maybe_subtract_poses(pose_keypoints, True)


if __name__ == '__main__':
    test_tumgaid()
    # test_openpose_utils()

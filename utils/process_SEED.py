import glob
import numpy as np
import scipy.io as sio


def normalization(data):
    '''
    description: 
    param {type} 
    return {type} 
    '''
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def get_data_per_subject(root, feature, label):
    feature_per_subject = None
    label_per_subject = None
    print(root)
    # load mat file
    data = sio.loadmat(root, verify_compressed_data_integrity=False)
    # Chosen feature from a subject

    for trial_idx in range(15):
        # transpose so that each row is a channel
        temp_data = normalization(
            data[feature + str(trial_idx+1)].transpose(1, 0, 2))
        num_of_sample = temp_data.shape[0]
        if feature_per_subject is None:
            feature_per_subject = temp_data
            label_per_subject = np.full(
                (num_of_sample, 1), label[0][trial_idx])
        else:
            feature_per_subject = np.vstack((feature_per_subject, temp_data))
            label_per_subject = np.vstack(
                (label_per_subject, np.full((num_of_sample, 1), label[0][trial_idx])))
        # 1s segmentation
        # shape of one trial->(100+,62,5)
    return feature_per_subject, label_per_subject


def get_data_label_from_mat(config, session_id):
    '''load raw data in SEED dataset with segmentation.

    Returns:
        sub_mov: DE feature from 15 moving clips.
        sub_ids: Smoothed DE feature.
        sub_label: Label for moving clips
    '''
    root = config['dataset_path']
    label = sio.loadmat(root+'label.mat')['label']
    print(root+str(session_id))
    mat_files = sorted(
        glob.glob(root+str(session_id)+'/*_*'))

    print('Total number of subjects: {:.0f}'.format(len(mat_files)))

    feature_all_subjects = []
    label_all_subjects = []
    for filepath in mat_files:
        # get every subject's trials
        feature_per_subject, label_per_subject = get_data_per_subject(
            filepath, config['feature'], label)
        feature_all_subjects.append(feature_per_subject)
        label_all_subjects.append(label_per_subject)
    return np.array(feature_all_subjects), np.array(label_all_subjects)


if __name__ == '__main__':
    get_data_label_from_mat()

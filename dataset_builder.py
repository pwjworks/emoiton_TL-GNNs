import numpy as np
import os

from utils.process_SEED import get_data_label_from_mat
from emotion_dataset import EmotionDataset


def to_categorical(y, num_classes=None, dtype='float32'):
    '''Generate label according to y

    Args:
        y (array): Label array which contains -1, 1, 0
        num_classes (int, optional): Classes amout. Defaults to None.
        dtype (str, optional): data type. Defaults to 'float32'.

    Returns:
        categorical: label
    '''
    # one-hot encoding
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def build_SEED_dataset(config):
    '''build dataset files from SEED mat files.

    Args:
        subjects (int): subjects amount
    '''

    # path to feature directories
    root = config['processed_path'] + \
        '{:s}/{:s}/'.format(config['dataset'], config['feature'])
    # build train dataset and test dataset for 15 subjects
    for session_id in range(1, 4):
        data, label = get_data_label_from_mat(config, session_id)
        subject_ids = list(range(config['subjects']))
        for test_index in subject_ids:
            train_index = subject_ids.copy()
            del train_index[test_index]

            X = data[train_index].reshape(-1, 62, 5)
            Y = label[train_index].reshape(-1)

            testX = data[test_index]
            testY = label[test_index]

            # get labels
            Y = to_categorical(Y, 3)
            testY = to_categorical(testY, 3)
            EmotionDataset(
                config, 'Train_session'+str(session_id), root, test_index, None, X, Y)
            EmotionDataset(
                config, 'Test_session'+str(session_id), root, test_index, None, testX, testY)


def get_dataset(config, session_id, subject_index):

    root = config['processed_path']+'{:s}/{:s}'.format(
        config['dataset'], config['feature'])
    path = root+'/processed/V_{:s}_{:s}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
        config['dataset'], config['feature'], 'Train_session'+str(session_id), config['subjects'], subject_index)
    print(path)
    if not os.path.exists(path):
        build_SEED_dataset(config, session_id)

    train_dataset = EmotionDataset(config, 'Train_session'+str(session_id), root,
                                   subject_index, None)
    test_dataset = EmotionDataset(config, 'Test_session'+str(session_id), root,
                                  subject_index, None)
    return train_dataset, test_dataset

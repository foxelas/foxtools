# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    import fox_utils
else:
    from foxtools.src import fox_utils

# Image size should be multiple of 32
DEFAULT_HEIGHT = 32


def load_data(name=None, fold=None):
    # name options: 'full', 'test', 'train'
    if name is None:
        name = 'full'

    conf = fox_utils.parse_config()
    output_dir = conf['Directories']['OutputDir']
    dataset_name = conf['Data Settings']['Dataset']
    file_name = dataset_name + '_' + name + '.h5'
    folder_name = conf['Folder Names']['DatasetsFolderName']

    if fold is None:
        fpath = os.path.join(output_dir, dataset_name, folder_name, file_name)
    else:
        fpath = os.path.join(output_dir, dataset_name, folder_name, str(fold), file_name)

    print("Read from ", fpath)
    data_list, key_list, label_images = fox_utils.load_dataset(fpath, 'image')

    # Prepare input data
    cropped_data = fox_utils.center_crop_list(data_list, DEFAULT_HEIGHT, DEFAULT_HEIGHT, True)

    cropped_labels = fox_utils.center_crop_list(label_images, DEFAULT_HEIGHT, DEFAULT_HEIGHT)

    x_raw = np.array(cropped_data, dtype=np.float32)
    y = np.array(cropped_labels, dtype=np.float32)

    df = pd.DataFrame({'id': key_list, 'data': x_raw, 'label': y})
    df.set_index('id', inplace=True)

    return df


def load_and_extract_data(name=None, fold=None):
    df = load_data(name, fold)
    x_data, y_data, id_data = extract_from_df(df)
    return x_data, y_data, id_data


def extract_from_df(df):
    x_data = df.data
    y_data = df.label
    id_data = df.id
    return x_data, y_data, id_data


def display_dataset_and_labels_sequentially(x_data, y_data):
    for (x, y) in zip(x_data, y_data):
        fox_utils.show_display_image(x)
        fox_utils.show_image(y)


def load_train_test(fold=None):
    if fold is None:
        x_train, y_train, id_train = load_and_extract_data('train')
        x_test, y_test, id_test = load_and_extract_data('test')
    else:
        x_train, y_train, id_train = load_and_extract_data('train', fold)
        x_test, y_test, id_test = load_and_extract_data('test', fold)

    # display_dataset_and_labels_sequentially(x_train, y_train)
    return x_train, x_test, y_train, y_test, id_train, id_test


def get_train_test(test_size=0.1, random_state=42):
    df = load_data('full')
    image_list = df.data
    label_list = df.label
    x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size, random_state)
    print('Data samples -- xtrain: ', len(x_train), ', xtest: ', len(x_test))

    id_train = x_train.id
    id_test = x_test.id
    x_train = x_train.data
    x_test = x_test.data

    # display_dataset_and_labels_sequentially(x_train, y_train)
    return x_train, x_test, y_train, y_test, id_train, id_test


def show_label_montage(name=None):
    image_list, label_list, id_list = load_and_extract_data(name)
    filename = os.path.join(fox_utils.conf['Directories']['OutputDir'], fox_utils.conf['Data Settings']['Dataset'],
                            name + '-data-montage.jpg')
    fox_utils.show_montage(image_list, filename, 'srgb')
    filename = os.path.join(fox_utils.conf['Directories']['OutputDir'], fox_utils.conf['Data Settings']['Dataset'],
                            name + '-labels-montage.jpg')
    fox_utils.show_montage(label_list, filename, 'grey')

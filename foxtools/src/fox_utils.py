import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import cv2

dirSep = os.sep


######################### Path #########################

def get_filenames(folder, target='.mat'):
    orig_listdir = os.listdir(folder)
    orig_listdir = [os.path.join(folder, x) for x in orig_listdir if target in x]
    return orig_listdir


def makedir(fpath):
    try:
        os.mkdir(fpath)
    except OSError as error:
        print('folder exists')


######################### Messages #########################        
def not_implemented():
    print('Not yet implemented.')
    return


def not_supported(varname):
    print('Not supported [', varname, '].')
    return


######################### Config #########################
import configparser


def get_base_dir():
    cwd = os.getcwd()
    parts = cwd.split(dirSep)
    target_index = len(parts) - 1 - parts[::-1].index("foxtools")
    parts = parts[0: target_index]
    parts.insert(1, os.sep)
    base_dir = os.path.join(*parts)
    return base_dir


def get_module_path():
    module_path = os.path.join(get_base_dir(), "foxtools", "src")
    return module_path


def get_config_path():
    settings_file = os.path.join(get_base_dir(), "conf", "config.ini")
    return settings_file


def parse_config():
    settings_file = get_config_path()
    config = configparser.ConfigParser()
    config.read(settings_file, encoding='utf-8')
    print("Loading from settings " + settings_file)
    print("Sections: " + ', '.join(config.sections()))
    return config

def get_savedir():
    dir_name = os.path.join(conf['Directories']['OutputDir'], conf['Data Settings']['Dataset'])
    return dir_name


######################### Load #########################

from scipy.io import loadmat
import h5py
import mat73


###expected form of hsi data: Height x Width x Wavelengths

def load_from_h5(fname):
    val = h5py.File(fname, 'r')
    return val


def load_from_mat73(fname, varname=''):
    mat = mat73.loadmat(fname)
    val = mat[varname]
    return val


def load_from_mat(fname, varname=''):
    val = loadmat(fname)[varname]
    return val


def load_target_mat(fname, varname):
    hsi = load_from_mat73(fname + '_target.mat', varname)
    return hsi


def load_white_mat(fname, varname):
    hsi = load_from_mat73(fname + '_white.mat', varname)
    return hsi


def load_black_mat(fname, varname):
    hsi = load_from_mat73(fname + '_black.mat', varname)
    return hsi


def load_images(fpath):
    images = []
    for filename in os.listdir(fpath):
        img = cv2.imread(os.path.join(fpath, filename))
        if img is not None:
            images.append(img)
    return images


def load_label_images(fpath):
    img_list = load_images(fpath)
    rot_img_list = [np.transpose(labelImg, [1, 0, 2]) for labelImg in img_list]
    return rot_img_list


######################### Process #########################

def get_labels_from_mask(img_list):
    labels = []
    for img in img_list:
        if img.ndim > 2:
            gray_im = cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            gray_im = img
        # print("Min", np.min(np.min(grayIm)), "and Max ", np.max(np.max(grayIm)))
        (thresh, blackAndWhiteImage) = cv2.threshold(gray_im, 170, 255, cv2.THRESH_BINARY)
        label_img = np.logical_not(blackAndWhiteImage)
        label_img = label_img.astype(np.int8)
        # plt.imshow(labelImg, cmap='gray')
        # plt.show()
        labels.append(label_img)
    return labels


def center_crop(image_data, target_height=None, target_width=None):
    width = image_data.shape[1]
    height = image_data.shape[0]

    if target_width is None:
        target_width = min(width, height)

    if target_height is None:
        target_height = min(width, height)

    left = int(np.ceil((width - target_width) / 2))
    right = width - int(np.floor((width - target_width) / 2))

    top = int(np.ceil((height - target_height) / 2))
    bottom = height - int(np.floor((height - target_height) / 2))

    if np.ndim(image_data) > 2:
        cropped_img = image_data[top:bottom, left:right, :]
    else:
        cropped_img = image_data[top:bottom, left:right]

    return cropped_img


def center_crop_list(data_list, target_height=64, target_width=64, show_image=False):
    cropped_data = []
    for x in range(len(data_list)):
        val = center_crop(data_list[x], target_width, target_height)
        cropped_data.append(val)

    if show_image:
        show_montage(cropped_data)

    return cropped_data


def load_image_dataset(fpath, sample_type='pixel'):
    f = load_from_h5(fpath)
    data_list = []
    label_list = []

    key_list = list(f.keys())

    for key_value in key_list:
        val = f[key_value]['image'][:]
        lab = f[key_value]['label'][:]
        # if val.shape[2] != 311 | val.shape[2] != 3:
        #    val = np.transpose(val, [1, 2, 0])
        data_list.append(val)
        label_list.append(lab.astype(np.int8))

    new_data_list = []
    if sample_type == 'pixel':
        new_data_list = flatten_images(data_list)
    elif sample_type == 'patch':
        not_implemented()
    elif sample_type == 'image':
        new_data_list = data_list
    else:
        not_supported('SampleType')
    return new_data_list, key_list, label_list


def flatten_image(hsi):
    return np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2])).transpose()


def flatten_images(image_list):
    X = [flatten_image(x) for x in image_list]
    stacked = np.concatenate(X, axis=1).transpose()
    return stacked


def patch_split(hsi, patch_dim=25):
    sb = np.array(hsi.shape)
    st = np.floor(sb[0:2] / patch_dim)
    cropped = center_crop(hsi, st[0] * patch_dim, st[1] * patch_dim)
    patchIndex = np.meshgrid(np.arange(0, st[0], dtype=np.int32), np.arange(0, st[1], dtype=np.int32))
    patchList = np.empty((int(st[0] * st[1]), patch_dim, patch_dim, hsi.shape[2]))
    i = 0
    for x, y in zip(patchIndex[0].flatten(), patchIndex[1].flatten()):
        # v = hsi[(0 + x*patchDim):(patchDim + x*patchDim), (0 + y*patchDim):(patchDim + y*patchDim),:]
        # print(np.max(v.flatten()))
        patchList[++i, :, :, :] = cropped[(0 + x * patch_dim):(patch_dim + x * patch_dim),
                                  (0 + y * patch_dim):(patch_dim + y * patch_dim), :]
        # print(np.max(patchList[i,:,:,:].flatten()))
    return patchList


###################### HSI ##################

def normalize_hsi(hsi_data, white, black):
    norm_hsi = (hsi_data - black) / (white - black + 0.0000001)
    return norm_hsi


######################### Reconstruct 3D #########################
import math


def xyz2rgb(imXYZ):
    d = imXYZ.shape
    r = math.prod(d[0:2])
    w = d[-1]
    XYZ = np.reshape(imXYZ, (r, w))

    M = [[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0414],
         [0.0557, -0.2040, 1.0570]]
    sRGB = np.transpose(np.dot(M, np.transpose(XYZ)))

    sRGB = np.reshape(sRGB, d)
    return sRGB


def get_display_image(hsi_image, image_type='srgb', channel=150):
    recon = []
    if image_type == 'srgb':
        [m, n, z] = hsi_image.shape
        if z == 3:
            colImage = np.reshape(hsi_image, (m * n, z))
            maxConst = np.max(colImage)
            minConst = np.min(colImage)
            colImage = (colImage - float(minConst)) / (float(maxConst) - float(minConst))
            recon = np.reshape(colImage, (m, n, 3))

        else:
            filename = os.path.join(get_base_dir(), "parameters", 'displayParam_311.mat')

            xyz = load_from_mat(filename, 'xyz')
            illumination = load_from_mat(filename, 'illumination')

            colImage = np.reshape(hsi_image, (m * n, z))
            normConst = np.amax(colImage)
            colImage = colImage / float(normConst)
            colImage = colImage * illumination
            colXYZ = np.dot(colImage, np.squeeze(xyz))

            imXYZ = np.reshape(colXYZ, (m, n, 3))
            imXYZ[imXYZ < 0] = 0
            imXYZ = imXYZ / np.amax(imXYZ)
            dispImage_ = xyz2rgb(imXYZ)
            dispImage_[dispImage_ < 0] = 0
            dispImage_[dispImage_ > 1] = 1
            dispImage_ = dispImage_ ** 0.4
            recon = dispImage_

    elif image_type == 'channel':
        recon = hsi_image[:, :, channel]

    elif image_type == 'grey':
        recon = hsi_image

    else:
        not_supported(image_type)

    return recon


######################### Plotting #########################
import skimage.util
import skimage.io


def simple_plot(y, fig_title, x_label, y_label, fpath):
    plt.plot(np.arange(len(y)) + 1, y)
    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt_filename = fpath + fig_title.replace(' ', '_') + '.jpg'
    print("Save figure at: ", plt_filename)
    plt.savefig(plt_filename)
    plt.show()


def show_display_image(hsi_image, image_type='srgb', channel=150):
    show_image(get_display_image(hsi_image, image_type, channel))


def show_image(x, fig_title=None, has_greyscale=False, fpath=""):
    if has_greyscale:
        plt.imshow(x, cmap='gray')
    else:
        plt.imshow(x)
    if fig_title is not None:
        plt.title(fig_title)
        plt_filename = os.path.join(fpath, fig_title.replace(' ', '_') + '.jpg')
        plt.savefig(plt_filename)
        print("Save figure at:" + plt_filename)
    plt.show()


def show_montage(data_list, filename=None, image_type='srgb', channel=150):
    # Needs to have same number of dimensions for each image, type float single

    hsiList = np.array([get_display_image(x, image_type, channel) for x in data_list], dtype='float64')
    if image_type != 'grey':
        m = skimage.util.montage(hsiList, channel_axis=3)
        m = (m * 255).astype(np.uint8)
    else:
        m = skimage.util.montage(hsiList)

    if filename is None:
        filename = os.path.join(conf['Directories']['OutputDir'], conf['Data Settings']['Dataset'],
                                'data-montage.jpg')
    skimage.io.imsave(filename, m)


############ Initialization #################
#print("Basedir:" + get_base_dir())
#print("Configdir:" + get_config_path())
conf = parse_config()

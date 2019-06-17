# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Gregoire Dauce (https://git.ee.ethz.ch/dauceg.student.ethz.ch)
# Robin Brügger

import os
import numpy as np
import logging
import string
import gc
import h5py
from skimage import transform

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5
alpha_dic = {ch: n for n, ch in enumerate(string.ascii_uppercase)}

def test_train_val_split(patient_id):

    if patient_id % 10 >= 8:
        return 'train'
    elif patient_id % 10 >= 6:
        return 'validation'
    else:
        return 'train'

def normalise_image(image):
    '''
    standardize based on nonzero pixels
    '''
    m = np.nanmean(np.where(image == 0, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
    s = np.nanstd(np.where(image == 0, np.nan, image), axis=(0,1,2)).astype(np.float32)
    normalized = np.divide((image - m), s)
    image = np.where(image == 0, 0, normalized)
    return image


def crop_volume_allDim(image, mask=None):
    '''
    Strip away the zeros on the edges of the three dimensions of the image
    Idea: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
    '''
    coords = np.argwhere(image > 0)
    x0, y0, z0, _ = coords.min(axis=0)
    x1, y1, z1, _ = coords.max(axis=0) + 1

    image = image[x0:x1, y0:y1, z0:z1, :]
    if not mask is None:
        return image, mask[x0:x1, y0:y1, z0:z1]
    return image

def crop_or_pad_slice_to_size(image, target_size, channels=None, offset=None):
    '''
    Make sure that the image has the desired dimensions
    '''
    
    if offset is None:
        offset = (0, 0, 0)
    
    x_t, y_t, z_t = target_size[0:3]
    x_s, y_s, z_s = image.shape[0:3]

    if not channels is None:
        output_volume = np.zeros((x_t, y_t, z_t, channels), dtype=np.float32)
    else:
        output_volume = np.zeros((x_t, y_t, z_t), dtype=np.float32)

    x_d = abs(x_t - x_s) // 2 + offset[0]
    y_d = abs(y_t - y_s) // 2 + offset[1]
    z_d = abs(z_t - z_s) // 2 + offset[2]

    t_ranges = []
    s_ranges = []

    for t, s, d in zip([x_t, y_t, z_t], [x_s, y_s, z_s], [x_d, y_d, z_d]):

        if t < s:
            t_range = slice(t)
            s_range = slice(d, d + t)
        else:
            t_range = slice(d, d + s)
            s_range = slice(s)

        t_ranges.append(t_range)
        s_ranges.append(s_range)

    if not channels is None:
        output_volume[t_ranges[0], t_ranges[1], t_ranges[2], :] = image[s_ranges[0], s_ranges[1], s_ranges[2], :]
    else:
        output_volume[t_ranges[0], t_ranges[1], t_ranges[2]] = image[s_ranges[0], s_ranges[1], s_ranges[2]]

    return output_volume

def prepare_data(input_folder, output_file, size, input_channels, target_resolution):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    if len(size) != 3:
        raise AssertionError('Inadequate number of size parameters')
    if len(target_resolution) != 3:
        raise AssertionError('Inadequate number of target resolution parameters')


    hdf5_file = h5py.File(output_file, "w")

    file_list = {'test': [], 'train': [], 'validation': []}

    logging.info('Counting files and parsing meta data...')

    pid = 0
    for folder in os.listdir(input_folder):
        print(folder)
        train_test = test_train_val_split(pid)
        pid = pid + 1
        file_list[train_test].append(folder)


    n_train = len(file_list['train'])
    n_test = len(file_list['test'])
    n_val = len(file_list['validation'])

    print('Debug: Check if sets add up to correct value:')
    print(n_train, n_val, n_test, n_train + n_val + n_test)

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            print([num_points] + list(size) + [input_channels])
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size) + [input_channels],
                                                              dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)
            data['pids_%s' % tt] = hdf5_file.create_dataset("pids_%s" % tt, [num_points] , dtype=h5py.special_dtype(vlen=str))

    mask_list = {'test': [], 'train': [], 'validation': []}
    img_list = {'test': [], 'train': [], 'validation': []}
    pids_list = {'test': [], 'train': [], 'validation': []}

    logging.info('Parsing image files')

    #get max dimension in z-axis
    maxX = 0
    maxY = 0
    maxZ = 0
    maxXCropped = 0
    maxYCropped = 0
    maxZCropped = 0
    i = 0
    for train_test in ['test', 'train', 'validation']:
        for folder in file_list[train_test]:
            print("Doing file {}".format(i))
            i += 1

            baseFilePath = os.path.join(input_folder, folder, folder)
            img_c1, _, img_header = utils.load_nii(baseFilePath + "_t1.nii.gz")
            img_c2, _, _ = utils.load_nii(baseFilePath + "_t1ce.nii.gz")
            img_c3, _, _ = utils.load_nii(baseFilePath + "_t2.nii.gz")
            img_c4, _, _ = utils.load_nii(baseFilePath + "_flair.nii.gz")
            img_dat = np.stack((img_c1, img_c2, img_c3, img_c4), 3)

            maxX = max(maxX, img_dat.shape[0])
            maxY = max(maxY, img_dat.shape[1])
            maxZ = max(maxZ, img_dat.shape[2])
            img_dat_cropped = crop_volume_allDim(img_dat)
            maxXCropped = max(maxXCropped, img_dat_cropped.shape[0])
            maxYCropped = max(maxYCropped, img_dat_cropped.shape[1])
            maxZCropped = max(maxZCropped, img_dat_cropped.shape[2])
    print("Max x: {}, y: {}, z: {}".format(maxX, maxY, maxZ))
    print("Max cropped x: {}, y: {}, z: {}".format(maxXCropped, maxYCropped, maxZCropped))

    for train_test in ['train', 'test', 'validation']:

        write_buffer = 0
        counter_from = 0

        for folder in file_list[train_test]:

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % folder)

            patient_id = folder

            baseFilePath = os.path.join(input_folder, folder, folder)
            img_c1, _, img_header = utils.load_nii(baseFilePath + "_t1.nii.gz")
            img_c2, _, _ = utils.load_nii(baseFilePath + "_t1ce.nii.gz")
            img_c3, _, _ = utils.load_nii(baseFilePath + "_t2.nii.gz")
            img_c4, _, _ = utils.load_nii(baseFilePath + "_flair.nii.gz")
            mask_dat, _, _ = utils.load_nii(baseFilePath + "_seg.nii.gz")

            img_dat = np.stack((img_c1, img_c2, img_c3, img_c4), 3)

            img, mask = crop_volume_allDim(img_dat.copy(), mask_dat.copy())

            pixel_size = (img_header.structarr['pixdim'][1],
                          img_header.structarr['pixdim'][2],
                          img_header.structarr['pixdim'][3])

            logging.info('Pixel size:')
            logging.info(pixel_size)

            ### PROCESSING LOOP FOR 3D DATA ################################

            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1],
                            pixel_size[2]/ target_resolution[2]]

            if scale_vector != [1.0, 1.0, 1.0]:
                img = transform.rescale(img, scale_vector, order=1, preserve_range=True, multichannel=True, mode='constant')
                mask = transform.rescale(mask, scale_vector, order=0, preserve_range=True, multichannel=False, mode='constant')

            img = crop_or_pad_slice_to_size(img, size, input_channels)
            mask = crop_or_pad_slice_to_size(mask, size)

            img = normalise_image(img)

            img_list[train_test].append(img)
            mask_list[train_test].append(mask)
            pids_list[train_test].append(patient_id)

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer
                _write_range_to_hdf5(data, train_test, img_list, mask_list, pids_list, counter_from, counter_to)
                _release_tmp_memory(img_list, mask_list, pids_list, train_test)

                # reset stuff for next iteration
                counter_from = counter_to
                write_buffer = 0

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        if len(file_list[train_test]) > 0:
            _write_range_to_hdf5(data, train_test, img_list, mask_list, pids_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, pids_list, train_test)

    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, pids_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr
    hdf5_data['pids_%s' % train_test][counter_from:counter_to, ...] = pids_list[train_test]


def _release_tmp_memory(img_list, mask_list, pids_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    mask_list[train_test].clear()
    pids_list[train_test].clear()
    gc.collect()



def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                input_channels,
                                target_resolution,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_3D_size_%s_res_%s.hdf5' % (size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, size, input_channels, target_resolution)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':
    input_folder = "path/to/original/training"
    preprocessing_folder = "path/to/desired/output/location"

    d = load_and_maybe_process_data(input_folder, preprocessing_folder, (160, 192, 160), 4, (1.0, 1.0, 1.0), force_overwrite=True)


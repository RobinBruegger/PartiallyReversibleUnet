import torch
import torch.utils.data
import h5py
import numpy as np
import time
import random
import dataProcessing.augmentation as aug

class BratsDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, filePath, expConfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False):
        super(BratsDataset, self).__init__()
        self.filePath = filePath
        self.mode = mode
        self.file = None
        self.trainOriginalClasses = expConfig.TRAIN_ORIGINAL_CLASSES
        self.randomCrop = randomCrop
        self.hasMasks = hasMasks
        self.returnOffsets = returnOffsets

        #augmentation settings
        self.nnAugmentation = expConfig.NN_AUGMENTATION
        self.softAugmentation = expConfig.SOFT_AUGMENTATION
        self.doRotate = expConfig.DO_ROTATE
        self.rotDegrees =  expConfig.ROT_DEGREES
        self.doScale = expConfig.DO_SCALE
        self.scaleFactor = expConfig.SCALE_FACTOR
        self.doFlip = expConfig.DO_FLIP
        self.doElasticAug = expConfig.DO_ELASTIC_AUG
        self.sigma = expConfig.SIGMA
        self.doIntensityShift = expConfig.DO_INTENSITY_SHIFT
        self.maxIntensityShift = expConfig.MAX_INTENSITY_SHIFT

    def __getitem__(self, index):

        #lazily open file
        self.openFileIfNotOpen()

        #load from hdf5 file
        image = self.file["images_" + self.mode][index, ...]
        if self.hasMasks: labels = self.file["masks_" + self.mode][index, ...]

        #Prepare data depeinding on soft/hard augmentation scheme
        if not self.nnAugmentation:
            if not self.trainOriginalClasses and (self.mode != "train" or self.softAugmentation):
                if self.hasMasks: labels = self._toEvaluationOneHot(labels)
                defaultLabelValues = np.zeros(3, dtype=np.float32)
            else:
                if self.hasMasks: labels = self._toOrignalCategoryOneHot(labels)
                defaultLabelValues = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)
        elif self.hasMasks:
            if labels.ndim < 4:
                labels = np.expand_dims(labels, 3)
            defaultLabelValues = np.asarray([0], dtype=np.float32)

        #augment data
        if self.mode == "train":
            image, labels = aug.augment3DImage(image,
                                               labels,
                                               defaultLabelValues,
                                               self.nnAugmentation,
                                               self.doRotate,
                                               self.rotDegrees,
                                               self.doScale,
                                               self.scaleFactor,
                                               self.doFlip,
                                               self.doElasticAug,
                                               self.sigma,
                                               self.doIntensityShift,
                                               self.maxIntensityShift)

        if self.nnAugmentation:
            if self.hasMasks: labels = self._toEvaluationOneHot(np.squeeze(labels, 3))
        else:
            if self.mode == "train" and not self.softAugmentation and not self.trainOriginalClasses and self.hasMasks:
                labels = self._toOrdinal(labels)
                labels = self._toEvaluationOneHot(labels)

        # random crop
        if not self.randomCrop is None:
            shape = image.shape
            x = random.randint(0, shape[0] - self.randomCrop[0])
            y = random.randint(0, shape[1] - self.randomCrop[1])
            z = random.randint(0, shape[2] - self.randomCrop[2])
            image = image[x:x+self.randomCrop[0], y:y+self.randomCrop[1], z:z+self.randomCrop[2], :]
            if self.hasMasks: labels = labels[x:x + self.randomCrop[0], y:y + self.randomCrop[1], z:z + self.randomCrop[2], :]

        image = np.transpose(image, (3, 0, 1, 2))  # bring into NCWH format
        if self.hasMasks: labels = np.transpose(labels, (3, 0, 1, 2))  # bring into NCWH format

        # to tensor
        #image = image[:, 0:32, 0:32, 0:32]
        image = torch.from_numpy(image)
        if self.hasMasks:
            #labels = labels[:, 0:32, 0:32, 0:32]
            labels = torch.from_numpy(labels) 

        #get pid
        pid = self.file["pids_" + self.mode][index]

        if self.returnOffsets:
            xOffset = self.file["xOffsets_" + self.mode][index]
            yOffset = self.file["yOffsets_" + self.mode][index]
            zOffset = self.file["zOffsets_" + self.mode][index]
            if self.hasMasks:
                return image, str(pid), labels, xOffset, yOffset, zOffset
            else:
                return image, pid, xOffset, yOffset, zOffset
        else:
            if self.hasMasks:
                return image, str(pid), labels
            else:
                return image, pid

    def __len__(self):
        #lazily open file
        self.openFileIfNotOpen()

        return self.file["images_" + self.mode].shape[0]

    def openFileIfNotOpen(self):
        if self.file == None:
            self.file = h5py.File(self.filePath, "r")

    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 3], dtype=np.float32)
        out[:, :, :, 0] = (labels != 0)
        out[:, :, :, 1] = (labels != 0) * (labels != 2)
        out[:, :, :, 2] = (labels == 4)
        return out

    def _toOrignalCategoryOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 5], dtype=np.float32)
        for i in range(5):
            out[:, :, :, i] = (labels == i)
        return out

    def _toOrdinal(self, labels):
        return np.argmax(labels, axis=3)

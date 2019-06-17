from comet_ml import Experiment, ExistingExperiment
import sys
sys.path.append("..")
import torch
import torch.optim as optim
import torch.nn as nn
import bratsUtils
import torch.nn.functional as F
import random

id = random.getrandbits(64)

#restore experiment
#VALIDATE_ALL = False
PREDICT = False
#RESTORE_ID = 395
#RESTORE_EPOCH = 350
#LOG_COMETML_EXISTING_EXPERIMENT = ""

#general settings
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "Nonreversible NO_NEW30"
EPOCHS = 1000
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

#hyperparameters
CHANNELS = 30
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

#logging settings
LOG_EVERY_K_ITERATIONS = 50 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = True
DO_INTENSITY_SHIFT = True
RANDOM_CROP = [128, 128, 128]

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

if LOG_COMETML:
    if not "LOG_COMETML_EXISTING_EXPERIMENT" in locals():
        experiment = Experiment(api_key="", project_name="", workspace="")
    else:
        experiment = ExistingExperiment(api_key="", previous_experiment=LOG_COMETML_EXISTING_EXPERIMENT, project_name="", workspace="")
else:
    experiment = None

#network funcitons
if TRAIN_ORIGINAL_CLASSES:
    loss = bratsUtils.bratsDiceLossOriginal5
else:
    #loss = bratsUtils.bratsDiceLoss
    def loss(outputs, labels):
        return bratsUtils.bratsDiceLoss(outputs, labels, nonSquared=True)


class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=False, secondConv=True, hasDropout=False):
        super(EncoderModule, self).__init__()
        groups = min(outChannels, CHANNELS)
        self.maxpool = maxpool
        self.secondConv = secondConv
        self.hasDropout = hasDropout
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, outChannels)
        if secondConv:
            self.conv2 = nn.Conv3d(outChannels, outChannels, 3, padding=1, bias=False)
            self.gn2 = nn.GroupNorm(groups, outChannels)
        if hasDropout:
            self.dropout = nn.Dropout3d(0.2, True)

    def forward(self, x):
        if self.maxpool:
            x = F.max_pool3d(x, 2)
        doInplace = INPLACE and not self.hasDropout
        x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=doInplace)
        if self.hasDropout:
            x = self.dropout(x)
        if self.secondConv:
            x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=INPLACE)
        return x

class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, upsample=False, firstConv=True):
        super(DecoderModule, self).__init__()
        groups = min(outChannels, CHANNELS)
        self.upsample = upsample
        self.firstConv = firstConv
        if firstConv:
            self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(groups, inChannels)
        self.conv2 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, outChannels)

    def forward(self, x):
        if self.firstConv:
            x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=INPLACE)
        x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=INPLACE)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x

class NoNewNet(nn.Module):
    def __init__(self):
        super(NoNewNet, self).__init__()
        channels = CHANNELS
        self.levels = 5

        self.lastConv = nn.Conv3d(channels, 3, 1, bias=True)

        #create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(4, channels, False, True, True))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i+1), True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), True, False))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, False))
        for i in range(self.levels - 2):
            decoderModules.append(DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        decoderModules.append(DecoderModule(channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x

net = NoNewNet()

optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, weight_decay=L2_REGULARIZER)
lr_sheudler = optim.lr_scheduler.MultiStepLR(optimizer, [250, 400, 550], 0.2)
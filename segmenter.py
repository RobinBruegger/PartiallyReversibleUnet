import torch
import torch.nn as nn
import time
import bratsUtils
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import dataProcessing.utils as utils
import systemsetup

class Segmenter:

    def __init__(self, expConfig, trainDataLoader, valDataLoader, challengeValDataLoader):
        self.expConfig = expConfig
        self.trainDataLoader = trainDataLoader
        self.valDataLoader = valDataLoader
        self.challengeValDataLoader = challengeValDataLoader
        self.experiment = expConfig.experiment
        self.checkpointsBasePathLoad = systemsetup.CHECKPOINT_BASE_PATH
        self.checkpointsBasePathSave= systemsetup.CHECKPOINT_BASE_PATH
        self.predictionsBasePath = systemsetup.PREDICTIONS_BASE_PATH
        self.startFromEpoch = 0

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0

        self.movingAvg = 0
        self.bestMovingAvg = 0
        self.bestMovingAvgEpoch = 1e9
        self.EXPONENTIAL_MOVING_AVG_ALPHA = 0.95
        self.EARLY_STOPPING_AFTER_EPOCHS = 120

        # restore model if requested
        if hasattr(expConfig, "RESTORE_ID") and hasattr(expConfig, "RESTORE_EPOCH"):
            self.startFromEpoch = self.loadFromDisk(expConfig.RESTORE_ID, expConfig.RESTORE_EPOCH) + 1
            print("Loading checkpoint with id {} at epoch {}".format(expConfig.RESTORE_ID, expConfig.RESTORE_EPOCH))

        # Run on GPU or CPU
        if torch.cuda.is_available():
            print("using cuda (", torch.cuda.device_count(), "device(s))")
            if torch.cuda.device_count() > 1:
                expConfig.net = nn.DataParallel(expConfig.net)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("using cpu")
        expConfig.net = expConfig.net.to(self.device)

    def validateAllCheckpoints(self):

        expConfig = self.expConfig

        print('==== VALIDATING ALL CHECKPOINTS ====')
        print(self.expConfig.EXPERIMENT_NAME)
        print("ID: {}".format(expConfig.id))
        print("RESTORE ID {}".format(expConfig.RESTORE_ID))
        print('====================================')

        for epoch in range(self.startFromEpoch, self.expConfig.EPOCHS):
            self.loadFromDisk(expConfig.RESTORE_ID, epoch)
            self.validate(epoch)

        #print best mean dice
        print("Best mean dice: {:.4f} at epoch {}".format(self.bestMeanDice, self.bestMeanDiceEpoch))

    def makePredictions(self):
        # model is already loaded from disk by constructor

        expConfig = self.expConfig
        assert(hasattr(expConfig, "RESTORE_ID"))
        assert(hasattr(expConfig, "RESTORE_EPOCH"))
        id = expConfig.RESTORE_ID
        epoch = expConfig.RESTORE_EPOCH

        print('============ PREDICTING ============')
        print(self.expConfig.EXPERIMENT_NAME)
        print("ID: {}".format(expConfig.id))
        print("RESTORE ID {}".format(expConfig.RESTORE_ID))
        print("RESTORE EPOCH {}".format(expConfig.RESTORE_EPOCH))
        print('====================================')

        basePath = os.path.join(self.predictionsBasePath, "{}_e{}".format(id, epoch))
        if not os.path.exists(basePath):
            os.makedirs(basePath)

        with torch.no_grad():
            for i, data in enumerate(self.challengeValDataLoader):
                inputs, pids, xOffset, yOffset, zOffset = data
                print("processing {}".format(pids[0]))
                print("Offsets: [{}, {}, {}]".format(xOffset, yOffset, zOffset))
                inputs = inputs.to(self.device)

                #predict labels and bring into required shape
                outputs = expConfig.net(inputs)
                outputs = outputs[:, :, :, :, :155]
                s = outputs.shape
                fullsize = outputs.new_zeros((s[0], s[1], 240, 240, 155))
                if xOffset + s[2] > 240:
                    outputs = outputs[:, :, :240-xOffset, :, :]
                if yOffset + s[3] > 240:
                    outputs = outputs[:, :, :, :240 - yOffset, :]
                if zOffset + s[4] > 155:
                    outputs = outputs[:, :, :, :, :155 - zOffset]
                fullsize[:, :, xOffset:xOffset+s[2], yOffset:yOffset+s[3], zOffset:zOffset+s[4]] = outputs
                print(fullsize.shape)

                #binarize output
                wt, tc, et = fullsize.chunk(3, dim=1)
                s = fullsize.shape
                wt = (wt > 0.5).view(s[2], s[3], s[4])
                tc = (tc > 0.5).view(s[2], s[3], s[4])
                et = (et > 0.5).view(s[2], s[3], s[4])

                result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
                result[wt] = 2
                result[tc] = 1
                result[et] = 4

                npResult = result.cpu().numpy()
                path = os.path.join(basePath, "{}.nii.gz".format(pids[0]))
                utils.save_nii(path, npResult, None, None)

        print("Done :)")

    def train(self):

        expConfig = self.expConfig
        expConfig.optimizer.zero_grad()

        print('======= RUNNING EXPERIMENT =======')
        print(self.expConfig.EXPERIMENT_NAME)
        print("ID: {}".format(expConfig.id))
        print('==================================')

        # for epoch in range(self.startFromEpoch, self.expConfig.EPOCHS):
        epoch = self.startFromEpoch
        while epoch < self.expConfig.EPOCHS and epoch <= self.bestMovingAvgEpoch + self.EARLY_STOPPING_AFTER_EPOCHS:

            running_loss = 0.0
            startTime = time.time()

            # set net up training
            self.expConfig.net.train()

            for i, data in enumerate(self.trainDataLoader):

                #load data
                inputs, pid, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #forward and backward pass
                outputs = expConfig.net(inputs)
                loss = expConfig.loss(outputs, labels)
                del inputs, outputs, labels
                loss.backward()

                #update params
                if i == len(self.trainDataLoader) - 1 or i % expConfig.VIRTUAL_BATCHSIZE == (expConfig.VIRTUAL_BATCHSIZE - 1):
                    expConfig.optimizer.step()
                    expConfig.optimizer.zero_grad()

                #logging every K iterations
                running_loss += loss.item()
                del loss
                if expConfig.LOG_EVERY_K_ITERATIONS > 0:
                    if i % expConfig.LOG_EVERY_K_ITERATIONS == (expConfig.LOG_EVERY_K_ITERATIONS - 1):
                        print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / expConfig.LOG_EVERY_K_ITERATIONS))
                        if expConfig.LOG_MEMORY_EVERY_K_ITERATIONS: self.logMemoryUsage()
                        running_loss = 0.0

            #logging at end of epoch
            if expConfig.LOG_MEMORY_EVERY_EPOCH: self.logMemoryUsage()
            if expConfig.LOG_EPOCH_TIME:
                print("Time for epoch: {:.2f}s".format(time.time() - startTime))
            if expConfig.LOG_LR_EVERY_EPOCH:
                for param_group in expConfig.optimizer.param_groups:
                    print("Current lr: {:.6f}".format(param_group['lr']))

            #validation at end of epoch
            if epoch % expConfig.VALIDATE_EVERY_K_EPOCHS == expConfig.VALIDATE_EVERY_K_EPOCHS - 1:
                self.validate(epoch)

            #take lr sheudler step
            if hasattr(expConfig, "lr_sheudler"):
                if isinstance(expConfig.lr_sheudler, optim.lr_scheduler.ReduceLROnPlateau):
                    expConfig.lr_sheudler.step(self.movingAvg)
                else:
                    expConfig.lr_sheudler.step()

            #save model
            #self.saveToDisk(epoch)

            epoch = epoch + 1

        #print best mean dice
        print("Best mean dice: {:.4f} at epoch {}".format(self.bestMeanDice, self.bestMeanDiceEpoch))

    def validate(self, epoch):

        #set net up for inference
        self.expConfig.net.eval()

        expConfig = self.expConfig
        hausdorffEnabled = (expConfig.LOG_HAUSDORFF_EVERY_K_EPOCHS > 0)
        logHausdorff = hausdorffEnabled and epoch % expConfig.LOG_HAUSDORFF_EVERY_K_EPOCHS == (expConfig.LOG_HAUSDORFF_EVERY_K_EPOCHS - 1)

        startTime = time.time()
        with torch.no_grad():
            diceWT, diceTC, diceET = [], [], []
            sensWT, sensTC, sensET = [], [], []
            specWT, specTC, specET = [], [], []
            hdWT, hdTC, hdET = [], [], []
            #buckets = np.zeros(5)

            for i, data in enumerate(self.valDataLoader):

                # feed inputs through neural net
                inputs, _, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = expConfig.net(inputs)

                if expConfig.TRAIN_ORIGINAL_CLASSES:
                    outputsOriginal5 = outputs
                    outputs = torch.argmax(outputs, 1)
                    #hist, _ = np.histogram(outputs.cpu().numpy(), 5, (0, 4))
                    #buckets = buckets + hist
                    wt = bratsUtils.getWTMask(outputs)
                    tc = bratsUtils.getTCMask(outputs)
                    et = bratsUtils.getETMask(outputs)

                    labels = torch.argmax(labels, 1)
                    wtMask = bratsUtils.getWTMask(labels)
                    tcMask = bratsUtils.getTCMask(labels)
                    etMask = bratsUtils.getETMask(labels)

                else:

                    #separate outputs channelwise
                    wt, tc, et = outputs.chunk(3, dim=1)
                    s = wt.shape
                    wt = wt.view(s[0], s[2], s[3], s[4])
                    tc = tc.view(s[0], s[2], s[3], s[4])
                    et = et.view(s[0], s[2], s[3], s[4])

                    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
                    s = wtMask.shape
                    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
                    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
                    etMask = etMask.view(s[0], s[2], s[3], s[4])

                #TODO: add special evaluation metrics for original 5

                #get dice metrics
                diceWT.append(bratsUtils.dice(wt, wtMask))
                diceTC.append(bratsUtils.dice(tc, tcMask))
                diceET.append(bratsUtils.dice(et, etMask))

                #get sensitivity metrics
                sensWT.append(bratsUtils.sensitivity(wt, wtMask))
                sensTC.append(bratsUtils.sensitivity(tc, tcMask))
                sensET.append(bratsUtils.sensitivity(et, etMask))

                #get specificity metrics
                specWT.append(bratsUtils.specificity(wt, wtMask))
                specTC.append(bratsUtils.specificity(tc, tcMask))
                specET.append(bratsUtils.specificity(et, etMask))

                #get hausdorff distance
                if logHausdorff:
                    lists = [hdWT, hdTC, hdET]
                    results = [wt, tc, et]
                    masks = [wtMask, tcMask, etMask]
                    for i in range(3):
                        hd95 = bratsUtils.getHd95(results[i], masks[i])
                        #ignore edgcases in which no distance could be calculated
                        if (hd95 >= 0):
                            lists[i].append(hd95)

        #calculate mean dice scores
        meanDiceWT = np.mean(diceWT)
        meanDiceTC = np.mean(diceTC)
        meanDiceET = np.mean(diceET)
        meanDice = np.mean([meanDiceWT, meanDiceTC, meanDiceET])
        if (meanDice > self.bestMeanDice):
            self.bestMeanDice = meanDice
            self.bestMeanDiceEpoch = epoch

        #update moving avg
        self._updateMovingAvg(meanDice, epoch)

        #print metrics
        print("------ Validation epoch {} ------".format(epoch))
        print("Dice        WT: {:.4f} TC: {:.4f} ET: {:.4f} Mean: {:.4f} MovingAvg: {:.4f}".format(meanDiceWT, meanDiceTC, meanDiceET, meanDice, self.movingAvg))
        print("Sensitivity WT: {:.4f} TC: {:.4f} ET: {:.4f}".format(np.mean(sensWT), np.mean(sensTC), np.mean(sensET)))
        print("Specificity WT: {:.4f} TC: {:.4f} ET: {:.4f}".format(np.mean(specWT), np.mean(specTC), np.mean(specET)))
        if logHausdorff:
            print("Hausdorff   WT: {:6.2f} TC: {:6.2f} ET: {:6.2f}".format(np.mean(hdWT), np.mean(hdTC), np.mean(hdET)))

        #log metrics
        if self.experiment is not None:
            self.experiment.log_metrics({"wt": meanDiceWT, "tc": meanDiceTC, "et":  meanDiceET, "mean": meanDice, "movingAvg": self.movingAvg}, "dice", epoch)
            self.experiment.log_metrics({"wt": np.mean(sensWT), "tc": np.mean(sensTC), "et": np.mean(sensET)}, "sensitivity", epoch)
            self.experiment.log_metrics({"wt": np.mean(specWT), "tc": np.mean(specTC), "et": np.mean(specET)}, "specificity", epoch)
            if logHausdorff:
                self.experiment.log_metrics({"wt": np.mean(hdWT), "tc:": np.mean(hdTC), "et": np.mean(hdET)}, "hausdorff", epoch)

        #print(buckets)

        #log validation time
        if expConfig.LOG_VALIDATION_TIME:
            print("Time for validation: {:.2f}s".format(time.time() - startTime))
        print("--------------------------------")


    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))


    def saveToDisk(self, epoch):

        #gather things to save
        saveDict = {"net_state_dict": self.expConfig.net.state_dict(),
                    "optimizer_state_dict": self.expConfig.optimizer.state_dict(),
                    "epoch": epoch,
                    "bestMeanDice": self.bestMeanDice,
                    "bestMeanDiceEpoch": self.bestMeanDiceEpoch,
                    "movingAvg": self.movingAvg,
                    "bestMovingAvgEpoch": self.bestMovingAvgEpoch,
                    "bestMovingAvg": self.bestMovingAvg}
        if hasattr(self.expConfig, "lr_sheudler"):
            saveDict["lr_sheudler_state_dict"] = self.expConfig.lr_sheudler.state_dict()

        #save dict
        path = self._getCheckpointPathSave(self.expConfig.id, epoch)
        torch.save(saveDict, path)

    def loadFromDisk(self, id, epoch):
        path = self._getCheckpointPathLoad(id, epoch)
        checkpoint = torch.load(path)
        self.expConfig.net.load_state_dict(checkpoint["net_state_dict"])

        #load optimizer: hack necessary because load_state_dict has bugs (See https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949)
        self.expConfig.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.expConfig.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        if "lr_sheudler_state_dict" in checkpoint:
            self.expConfig.lr_sheudler.load_state_dict(checkpoint["lr_sheudler_state_dict"])
            #Hack lr sheudle
            #self.expConfig.lr_sheudler.milestones = [250, 400, 550]

        #load best epoch score (if available)
        if "bestMeanDice" in checkpoint:
            self.bestMeanDice = checkpoint["bestMeanDice"]
            self.bestMeanDiceEpoch = checkpoint["bestMeanDiceEpoch"]

        #load moving avg if available
        if "movingAvg" in checkpoint:
            self.movingAvg = checkpoint["movingAvg"]

        #load best moving avg epoch if available
        if "bestMovingAvgEpoch" in checkpoint:
            self.bestMovingAvgEpoch = checkpoint["bestMovingAvgEpoch"]
        if "bestMovingAvg" in checkpoint:
            self.bestMovingAvg = checkpoint["bestMovingAvg"]

        return checkpoint["epoch"]

    def _getCheckpointPathSave(self, id, epoch):
        return self.checkpointsBasePathSave + "{}/e_{}.pt".format(id, epoch)

    def _getCheckpointPathLoad(self, id, epoch):
        return self.checkpointsBasePathLoad + "{}/e_{}.pt".format(id, epoch)

    def _updateMovingAvg(self, validationMean, epoch):
        if self.movingAvg == 0:
            self.movingAvg = validationMean
        else:
            alpha = self.EXPONENTIAL_MOVING_AVG_ALPHA
            self.movingAvg = self.movingAvg * alpha + validationMean * (1 - alpha)

        if self.bestMovingAvg < self.movingAvg:
            self.bestMovingAvg = self.movingAvg
            self.bestMovingAvgEpoch = epoch

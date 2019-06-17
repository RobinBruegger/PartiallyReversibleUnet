# Partially Reversible U-Net Code
Public PyTorch implementation for our paper [A Partially Reversible U-Net for Memory-Efficient Volumetric Image Segmentation](https://arxiv.org/abs/1906.06148),
which was accepted for presentation at [MICCAI 2019](https://www.miccai2019.org/). 

If you find this code helpful in your research please cite the following paper:
```
@article{PartiallyRevUnet2019Bruegger,
         author={Br{\"u}gger, Robin and Baumgartner, Christian F.
         and Konukoglu, Ender},
         title={A Partially Reversible U-Net for Memory-Efficient Volumetric Image Segmentation},
         journal={arXiv:1906.06148},
         year={2019},
```

If you want to create your own reversible or partially reversible neural network you can use our [RevTorch](https://github.com/RobinBruegger/RevTorch) library.

## Virtual Environment Setup
The code is implemented in Python 3.6 using PyTorch 1.1.0. Follow the steps below to install all dependencies:
* Set up a virtual environment (e.g. conda or virtualenv) with Python 3.6
* Install all non-PyTorch requirements using: `pip install -r dataProcessing/brats18_data_loader.py`
* Install PyTorch following the instructions on ther [website](https://pytorch.org/).

## Data
We trained with the [BraTS 2018 dataset](https://www.med.upenn.edu/sbia/brats2018/data.html), which is available from the organizers of the [BraTS challenge](https://www.med.upenn.edu/sbia/brats2018.html).

To prepare the data, adjust the paths at the end of `dataProcessing/brats18_data_loader.py`. Then run this script. Do the same for `dataProcessing/brats18_validation_data_loader.py`, which prepares the validation data.
	
## Running the code
* Adjust the path in `systemsetup.py` to match your system.
* Run the `train.py` script

The settings for the experiments are each in an individual file located in the `experiments/` folder.
You can change the experiment by importing a different experiment in the file `segmenter.py`.

## Creating checkpoits and prediction
* To create a checkpoint after every epoch, set the `SAVE_CHECKPOINTS = True` in the corresponding experiment file
* For inference, you need to load a checkpoint form a previously trained model. To achieve this, set the following three fields in the corresponding experiment file
```
PREDICT = True
RESTORE_ID = <Id to load>
RESTORE_EPOCH = <Epoch to load>
```
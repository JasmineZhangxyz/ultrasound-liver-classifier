# UofT Capstone &lt;> Oncoustics: Automatic liver detection in ultrasound images

This project performs liver detection for Clarius transducer ultrasound scans using three different methods: SVM, CNN and Pretrained Models. 

### Authors
- Willis Guo ()
- Sophie Lee ()
- Allan Ma ()
- Nicole Streltsov ([@NicStrel](https://github.com/NicoleStrel])) 
- Jasmine Zhang ([@JasmineZhangxyz](https://github.com/JasmineZhangxyz))

## Local Running


To download the required dependencies,

- Create an environment via `python -m venv venv`
- Active the environment with `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (MacOS, UNIX)
- Install dependencies with `pip install requirements.txt`

Note that this project is highly reliant on using CUDA (GPU) - recommended.
To download the data necessary for training and evaluating, download the .npz file from [here](https://drive.google.com/file/d/13zsG_NFNqz4I6AA0cOJTdWWsr4bWrBr3/view?usp=sharing) and upload it to `(this repository)/data/` folder. 

## Data

#### We investigated three types of datasets: 
1. Self-collected Clarius Transducer scans (~7K scans)
2. Augmented self-collected Clarius Transducer scans (~30K scans)
3. Augmented self-collected Clarius Transducer scans with Augmented External datasets. (~ 35K scans)
   
#### Data Augmentation

We performed translation, rotation, cropping, contrast, Gaussian noise - where the Gaussian function applies variations on pixel intensity, and Gaussian blur - which performs smoothing by convolving a Gaussian kernel on all ultrasound scan images. 

#### External Datasets 

- *B-mode fatty liver ultrasound dataset* contains liver scans from 55 severely obese patients **(liver samples)**
- *Organ classification on abdominal ultrasound* contains labelled B-mode ultrasounds of various abdominal organs: liver, kidney, gallbladder, bowel, bladder, and spleen. **(liver and non-liver samples)**
- *COVID-19 lung ultrasound dataset* contains lung B-mode ultrasounds of various COVID-19 patients **(non-liver samples)**

#### Files

- `data_preprocessing.py`: defines a general function to load data, with the option of using external datasets and augmenting the data.  

## Methods

- `Liver_SVM.ipynb`: 
- `Liver_CNN.ipynb`: Simple CNN with 2 Convolutional Layers (with ReLU activation, 2x2 Max-Pooling) with 3x3 kernel, 30% Dropout Layer, and 2 Fully-Connected Layers. We use an Adam optimizer and Cross Entropy Loss function for training. This Notebook also includes attention map visualizations to identify how the model recognizes liver features. 

- the `/models` folder holds saved models for loading.

## Helper Functions

- `utility.py`: Defines helper functions for memory/runtime calculation, and metrics (AUC-ROC curve, confusion matrix plotting)

  







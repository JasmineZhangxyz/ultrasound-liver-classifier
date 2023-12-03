# UofT Capstone &lt;> Oncoustics: Automatic liver detection in ultrasound images

This project performs liver detection for Clarius transducer ultrasound scans using three different methods: SVM, CNN and Pretrained Models. 

### Authors
- Willis Guo ()
- Sophie Lee ()
- Allan Ma ()
- Nicole Streltsov ([@NicStrel](https://github.com/NicoleStrel])) 
- Jasmine Zhang ([@JasmineZhangxyz](https://github.com/JasmineZhangxyz))

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

## Helper Functions

- `utility.py`: Defines helper functions for memory/runtime calculation, and metrics (AUC-ROC curve, confusion matrix plotting)

  







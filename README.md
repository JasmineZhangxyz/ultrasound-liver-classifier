# UofT Capstone &lt;> Oncoustics: Automatic liver detection in ultrasound images

This project performs liver detection for Clarius transducer ultrasound scans using three different methods: SVM, CNN and Pretrained Models. 

### Authors
- Willis Guo ()
- Sophie Lee ([@lee14334](https://github.com/lee14334]))
- Allan Ma ([@allan-ma1](https://github.com/allan-ma1))
- Nicole Streltsov ([@NicStrel](https://github.com/NicoleStrel])) 
- Jasmine Zhang ([@JasmineZhangxyz](https://github.com/JasmineZhangxyz))

## Local Running


To download the required dependencies,

- Create an environment via `python -m venv venv`
- Active the environment with `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (MacOS, UNIX)
- Install dependencies with `pip install requirements.txt`

Note that this project is highly reliant on using CUDA (GPU) - recommended.
To download the data necessary for training and evaluating, download the all_data.npz file from the Google Drive Link Provided and upload it to `(this repository)/data/` folder. 

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
- `create_all_data_npz.py`: shows how we created all_data.npz file that is used in each of the methods notebooks. We downloaded the Clarius transducer scans in zip files and extracted images from those zip files. We excluded forearms and ensured from the collected data that 4 patients were dedicated to training, 1 for validation, and 1 for test sets. 

## Methods

- `SVM.ipynb`: SVM with an RBF kernel. We first normalize the data and apply PCA (retain 70% of the variance) to reduce dimensionality before training our SVM model. 
- `Liver_CNN.ipynb`: Simple CNN with 2 Convolutional Layers (with ReLU activation, 2x2 Max-Pooling) with 3x3 kernel, 30% Dropout Layer, and 2 Fully-Connected Layers. We use an Adam optimizer and Cross Entropy Loss function for training. This Notebook also includes attention map visualizations to identify how the model recognizes liver features. 
- `TransferLearning.ipynb`: 4 pre-trained models (ResNet-18, GoogLeNet, AlexNet, VGGNet-16) with frozen weights. A custom classification layer is added and trained for the purposes of transfer learning.

- the `/models` folder holds saved models for loading.
    - Note that for the pre-trained models, we only saved resnet and googlenet due to GitHub space limitations.

## Metrics

- `utility.py`: Defines helper functions for memory/runtime calculation, and metrics (AUC-ROC curve, confusion matrix plotting)
- `Comparisons.ipynb`: Creates the overall AUC-ROC curve and accuracy/runtime heatmap for the augmented dataset for the SVM, CNN and AlexNet (best pre-trained model)

  







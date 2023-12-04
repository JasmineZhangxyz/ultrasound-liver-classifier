import zipfile
from concurrent.futures import ThreadPoolExecutor
import io
import numpy as np
from PIL import Image, ImageOps
from functools import partial
import threading

def process_single_image(zip_path, img_size, f, result, lock):
    '''
    read each image and convert to numpy array

    :@param zip_path (string) zip file path
    :@param img_size (tuple) desired size of image
    :@param f - file object
    :@return image numpy array
    '''
    with zipfile.ZipFile(zip_path, 'r') as zip:
        with zip.open(f) as img_file:
            img = Image.open(io.BytesIO(img_file.read()))
            img = ImageOps.grayscale(img)
            img = img.resize(img_size)
            with lock: #only append if retrieve lock
                result.append(np.array(img))

def images_from_zip_path(zip_path, img_size, exclude = ''):
    '''
    Uses multithreading to concurrently unzip and convert images to grayscale numpy arrays (resized to img_size).
    Removes files that are in the exclude folder.

    :@param zip_path (string) zip file path
    :@param img_size (tuple) desired size of image
    :@param exclude (string) folder to exclude 
    :@return numpy array of images
    
    '''

    images = []
    lock = threading.Lock()

    # unzip the zip path
    with zipfile.ZipFile(zip_path, 'r') as zip:
        file_images_list = zip.namelist()

        #c reate thread for each image
        threads = []
        for f in file_images_list:
            if not(exclude) or not(f.startswith(exclude)):
                thread = threading.Thread(target=process_single_image, args=(zip_path, img_size, f, images, lock))
                threads.append(thread)
                thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    if not images:
        raise ValueError("No images found.")

    return np.stack(images, axis=0)

def get_seperated_data(zip_path, img_size, test_liver_name = '', val_liver_name = ''):
    '''
    Uses multithreading to concurrently unzip and convert images to grayscale numpy arrays (resized to img_size).
    Seperate images from two different folders in tow variables from one zip file.

    :@param zip_path (string) zip file path
    :@param test_liver_name (string) name of first folder
    :@param val_liver_name (string) name of second folder
    :@return test_liver, val_liver numpy arrays
    '''

    test_liver = []
    val_liver = []
    lock = threading.Lock()

    # unzip the zip path
    with zipfile.ZipFile(zip_path, 'r') as zip:
        file_images_list = zip.namelist()
        
        # Create a thread for each image
        threads = []
        for f in file_images_list:
            if f.startswith(test_liver_name):
                thread = threading.Thread(target=process_single_image, args=(zip_path, img_size, f, test_liver, lock))
                threads.append(thread)
                thread.start()
            elif f.startswith(val_liver_name):
                thread = threading.Thread(target=process_single_image, args=(zip_path, img_size, f, val_liver, lock))
                threads.append(thread)
                thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    if not (test_liver or val_liver):
        raise ValueError("No images found.")

    return np.stack(test_liver, axis=0), np.stack(val_liver, axis=0)

def balance_data(liver, non_liver):
    '''
    Balance liver and non-liver data

    :@param liver (numpy array)
    :@param liver (numpy array)
    '''

    n_liver = liver.shape[0]
    n_non_liver = non_liver.shape[0]

    if n_liver > n_non_liver:
        indices = np.random.choice(n_liver, n_non_liver, replace=False)
        liver = liver[indices]
    else:
        indices = np.random.choice(n_non_liver, n_liver, replace=False)
        non_liver = non_liver[indices]

    return liver, non_liver

def create_all_data_npz(data_dir):
    '''
    Creates all_data.npz as the main data source for all methods.

    :@param data_dir (string) directory for all zip files (downloads from Clarius cloud)
    '''

    img_size = (128, 128)
   
    # get non-liver data (self-collected, 4 patients) and exclude forearm scans
    print("Non-liver self-collected data")
    exclude_forearm = 'Oncoustics_Liver Exam - 07-Nov-2023_219_PM' # forearm scans
    non_liver = images_from_zip_path(data_dir + 'non-liver.zip', img_size, exclude = exclude_forearm)
    
    # get liver data (self-collected, 4 patients)
    print("Liver self-collected data")
    liver = images_from_zip_path(data_dir + 'liver.zip', img_size)

    # # get new liver data (collected by Oncoustics)
    print("Additional collected data")
    test_liver_name = 'Oncoustics_Liver Exam - 23-Nov-2023_126_PM'
    val_liver_name = 'Oncoustics_Liver Exam - 23-Nov-2023_111_PM'
    test_liver, val_liver = get_seperated_data(data_dir + 'new_liver.zip', img_size, test_liver_name, val_liver_name)
    
    # ------- external datasets -----
    print("External datasets")

    # CLF liver dataset
    clf_liver = images_from_zip_path(data_dir + 'clf_liver.zip', img_size)
    clf_non_liver = images_from_zip_path(data_dir + 'clf_non_liver.zip', img_size)
    clf_liver, clf_non_liver = balance_data(clf_liver, clf_non_liver) # balance the data
    
    # fatty liver dataset
    fatty_liver = images_from_zip_path(data_dir +'fatty_liver.zip', img_size)
    
    # lung dataset
    lung = images_from_zip_path(data_dir + 'lung.zip', img_size)

    # ------ construct dataset ------
    data_dict = {
        'liver': liver,
        'test_liver': test_liver,
        'val_liver': val_liver,
        'non_liver': non_liver,
        'clf_liver': clf_liver,
        'clf_non_liver': clf_non_liver,
        'fatty_liver': fatty_liver,
        'lung': lung
    }

    print(liver.shape, test_liver.shape, val_liver.shape, non_liver.shape, clf_liver.shape, clf_non_liver.shape, fatty_liver.shape, lung.shape)

    np.savez(data_dir + 'all_data.npz', **data_dict)

if __name__ == "__main__":
    data_dir = './data/'
    create_all_data_npz(data_dir)
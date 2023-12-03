
import numpy as np
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
import imgaug as ia
from sklearn.model_selection import train_test_split

def augment_data(liver, non_liver, translation=True, rotation=True, cropping=True, noise=True, blurring=True, contrast=True):
  '''
  Applies singular data augmentation techniques to input image datasets

  :@param liver, non_liver numpy array data
  :@param translation, rotation, cropping, noise, blurring, contrast (bool) transformation methods
  :@return augmented_liver, augmented_non_liver numpy arrays
  '''

  ia.seed(4)
  combined_liver = {"initial":liver}
  combined_non_liver = {"initial":non_liver}

  # translate
  if translation:
    translate = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    combined_liver['translated_liver'] = translate(images=liver)
    combined_non_liver['translated_non_liver'] = translate(images=non_liver)

  # rotate
  if rotation:
    rotate = iaa.Affine(rotate=(-25, 25))
    combined_liver['rotated_liver'] = rotate(images=liver)
    combined_non_liver['rotated_non_liver'] = rotate(images=non_liver)

  # scale+crop
  if cropping:
    crop = iaa.Crop(percent=(0, 0.2))
    combined_liver['cropped_liver'] = crop(images=liver)
    combined_non_liver['cropped_non_liver'] = crop(images=non_liver)

  # Gaussian noise
  if noise:
    gnoise = iaa.AdditiveGaussianNoise(scale=(10, 30))
    combined_liver['gnoisy_liver'] = gnoise(images=liver)
    combined_non_liver['gnoisy_non_liver'] = gnoise(images=non_liver)

  # Gaussian blur
  if blurring:
    gblur = iaa.GaussianBlur(sigma=(0.3, 0.7))
    combined_liver['gblur_liver'] = gblur(images=liver)
    combined_non_liver['gblur_non_liver'] = gblur(images=non_liver)

  # contrast
  if contrast:
    contrasted = iaa.LinearContrast((0.75, 1.5))
    combined_liver['contrast_liver'] = contrasted(images=liver)
    combined_non_liver['contrast_non_liver'] = contrasted(images=non_liver)

  # combine
  augmented_liver = np.concatenate(list(combined_liver.values()), axis=0)
  augmented_non_liver = np.concatenate(list(combined_non_liver.values()), axis=0)

  return augmented_liver, augmented_non_liver

def get_dataset(use_external=False, aug_data=False):
    '''
    Main function to construct the dataset based on options.

    :@param use_external (bool): Uses external dataset ultrasound scans
    :@param aug_data (bool): Augment the data
    :@return x_train, x_val, x_test, y_train, y_val, y_test numpy arrays
    '''

    data_dict = np.load('./data/all_data.npz')

    liver = data_dict['liver']
    val_liver = data_dict['val_liver']
    test_liver =  data_dict['test_liver']

    non_liver = data_dict['non_liver']
    non_liver, test_non_liver = train_test_split(non_liver, test_size=test_liver.shape[0], random_state=41)
    non_liver, val_non_liver = train_test_split(non_liver, test_size=val_liver.shape[0], random_state=41)

    if use_external:
        liver = np.concatenate([liver, data_dict['clf_liver'], data_dict['fatty_liver']], axis=0)
        non_liver = np.concatenate([non_liver, data_dict['clf_non_liver'], data_dict['lung']], axis=0)

    if aug_data:
        liver, non_liver = augment_data(liver, non_liver)

    x_train = np.concatenate([liver, non_liver], axis=0)
    x_val = np.concatenate([val_liver, val_non_liver], axis=0)
    x_test = np.concatenate([test_liver, test_non_liver], axis=0)

    y_train = np.concatenate([np.ones(liver.shape[0]), np.zeros(non_liver.shape[0])])
    y_val = np.concatenate([np.ones(val_liver.shape[0]), np.zeros(val_non_liver.shape[0])])
    y_test = np.concatenate([np.ones(test_liver.shape[0]), np.zeros(test_non_liver.shape[0])])

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)

    return x_train, x_val, x_test, y_train, y_val, y_test

import pandas as py
import numpy as np
import kaggle as kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import dlib
from PIL import Image
import os
import cv2
from deepface import DeepFace

def has_face_deepface(image_path):
    """
    Checks if an image contains a face using the deepface library.
    
    Parameters
    ----------
    image_path : str
        The path to the image to check.
    
    Returns
    -------
    bool
        True if the image contains a face, False otherwise.
    """
    try:
        analysis = DeepFace.extract_faces(image_path, detector_backend='opencv')
        return True
    except Exception as e:
        return False

#uses API  from my kaggle account in directory .kaggle cotnaning my info 
# in kaggle.json
#api = KaggleApi()
#api.authenticate()
api = KaggleApi()
api.authenticate()
potdata = ['ismailpromus/skin-diseases-image-dataset', 
           "shakyadissanayake/oily-dry-and-normal-skin-types-dataset",
           "amellia/face-skin-disease",
           "andreibadescu10/skin-diseases-dataset-3",
           "syedalinaqvi/augmented-skin-conditions-image-dataset",
           "pacificrm/skindiseasedataset",]
keywords = ['eczema','rosacea',"acne","healthy", "clear", "oily", "dry", "sun"]
total = 0
for dataset in potdata:
        files = api.dataset_list_files(dataset).files
        target_files = [
            file.name for file in files
            if any(keyword.lower() in file.name.lower() for keyword in keywords)
                #if "test" in file.name:
        ]
        print(target_files)
        total += len(target_files)
        #if target_files:
print("len is" ,total)





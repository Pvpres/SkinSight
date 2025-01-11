#import pandas as py
#import numpy as np
import kaggle as kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import dlib
#from PIL import Image
import os
import cv2
from deepface import DeepFace
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor

def has_face_dlib(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    return len(faces) > 0

def process_file(filepath, target, filename):
    # Processes a single file: checks for a face and moves 
    # it if a face is detected.
    if os.path.isfile(filepath) and has_face_dlib(filepath):
        shutil.move(filepath, os.path.join(target, filename))
        return True
    return False
def download_and_filter(api, dataset, directory):
    #creates directory for dataset using name of dataset
    dataset_dir = os.path.join(directory, dataset.split("/")[-1])
    if os.path.exists(dataset_dir):
        print(f"Dataset {dataset} already exists")
        return
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Processing data in: {dataset}")
    #creates zip path for dataset
    zip_path = os.path.join(dataset_dir, f"{dataset.split('/')[-1]}.zip")
    #downloads dataset as zipfile for speed and storage reasons (only one HTTP request instead of 100k+)
    api.dataset_download_files(dataset, path=dataset_dir, unzip=False)
    #extracts contents from zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    #removes zipfile
    matches = api.dataset_list_files(dataset)
    os.remove(zip_path)
    return matches

#directory to read from
#keyword to look for
#new folder to move files to
def sort_files(directory, keyword, folder):
    #where we want folder to be a folder named after keyword
    if(keyword == "healthy" or keyword == "clear" or keyword == "normal" 
       or keyword == "pins"):
        keyword = "normal"
    target = os.path.join(os.getcwd(),folder, keyword)
    os.makedirs(target, exist_ok=True)
    #contains all filepaths that will be moved
    filepaths = []
    #goes through all directories etc in given directory
    for root, dirs, files in os.walk(directory):
        #subdirectory in dirs
        for subdir in dirs:
            #if subdirectory has keyword
            if keyword in subdir.lower():
                #creates path from root to subdirectory
                subpath = os.path.join(root, subdir)
                
                for filename in os.listdir(subpath):
                    filepath = os.path.join(subpath, filename)
                    if os.path.isfile(filepath):
                        filepaths.append((filepath, target, filename))
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(lambda args: process_file(*args), filepaths))
    moved = sum(results)
    print(f"Moved {moved} {keyword} images to {folder}")
        
def preprocess (datasets, keywords, directory):
    #uses API  from my kaggle account in directory .kaggle cotnaning my info 
    # in kaggle.json
    api = KaggleApi()
    api.authenticate()
    #all datasets i will be using
    #keywords for datasets to use
    #only downloads datasets that have these keywords
    #keywords = ['eczema','rosacea',"acne","healthy", "clear", "oily", "dry", "sun"]
    total = 0
    #for each datasat in list it downloads and fiter files
    for dataset in datasets:
        matching = download_and_filter(api, dataset, directory)
        if hasattr(matching, 'files'):
            for file in matching.files:
                total += 1
        else:
            print(f"Dataset {dataset} has no iterator")
    print(f"Total number of usable images: {total}")
    for keyword in keywords:
        sort_files(directory, keyword, "usable_data")
    shutil.rmtree(directory)
    return total

if __name__ == "__main__":
    potdata = ['ismailpromus/skin-diseases-image-dataset',"shakyadissanayake/oily-dry-and-normal-skin-types-dataset",
               "amellia/face-skin-disease",
               "andreibadescu10/skin-diseases-dataset-3",
               "syedalinaqvi/augmented-skin-conditions-image-dataset",
               "pacificrm/skindiseasedataset", 
               "hereisburak/pins-face-recognition"]
    keywords = ['eczema','rosacea',"acne","healthy", "clear", "oily", "dry", "sun", "normal", "pins"]
    preprocess(potdata, keywords, "train_data")
    
               


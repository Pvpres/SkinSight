import kaggle as kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import cv2
import zipfile
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
# Function to normalize data (resize to 128x128 and center on face)
def preprocess_image(image_path, target, filename):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}", file=sys.stderr)
        return False

    # Convert to grayscale and detect face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return False
    x, y, w, h = faces[0]  # OpenCV rectangle format: (x, y, width, height)
    face_crop = image[y:y + h, x:x + w]
    
    # Resize the cropped face to 128x128
    resized_face = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_AREA)
    output_path = os.path.join(target, filename)
    cv2.imwrite(output_path, resized_face)
    return True

# Function to process files in parallel
def process_files_parallel(filepaths, target):
    """
    Processes files in parallel using ProcessPoolExecutor.
    """
    print("Processing files in parallel", file=sys.stderr)
    moved = 0
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [executor.submit(preprocess_image, filepath, target, filename)
                   for filepath, target, filename in filepaths]
        
        # Collect results as they complete
        for future in futures:
            if future.result():
                moved += 1
    return moved

#directory to read from
#keyword to look for
#new folder to move files to
def sort_files(directory, keyword, folder):
    #where we want folder to be a folder named after keyword
    if(keyword in  ["healthy","clear","normal","pins"]):
        keyword = "normal"
    print(f"Sorting {keyword} images", file=sys.stderr)
    target = os.path.join(os.getcwd(),folder, keyword)
    os.makedirs(target, exist_ok=True)
    #contains all filepaths that will be moved
    filepaths = []
    #goes through all directories etc in given directory
    for root, dirs, files in os.walk(directory):
        if keyword in root.lower():
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg",".bmp", ".tiff", ".webp")):
                    filepath = os.path.join(root, filename)
                    filepaths.append((filepath, target, filename))
    print(f"Found {len(filepaths)} files to process for keyword: {keyword}")
    #uses multithreading to process files
    moved = process_files_parallel(filepaths, target)
    print(f"Moved {moved} {keyword} images to {folder}")
    return moved

def download_and_filter(api, dataset, directory):
    #creates directory for dataset using name of dataset
    dataset_dir = os.path.join(directory, dataset.split("/")[-1])
    #checks if dataset already exists
    if os.path.exists(dataset_dir):
        print(f"Dataset {dataset} already exists")
        return
    #creates directory
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
    os.remove(zip_path)
    return

def preprocess (datasets, keywords, directory):
    #uses API  from my kaggle account in directory .kaggle cotnaning my info 
    # in kaggle.json
    api = KaggleApi()
    api.authenticate()
    #all datasets I will be using
    #keywords for datasets to use
    #only downloads datasets that have these keywords
    #keywords = ['eczema','rosacea',"acne","healthy", "clear", "oily", "dry", "sun"]
    #for each datasat in list it downloads and fiter files
    for dataset in datasets:
        download_and_filter(api, dataset, directory)
    num = 0
    for keyword in keywords:
        num += sort_files(directory, keyword, "usable_data")
    print(f"Total number of face-confirmed images: {num}", file=sys.stderr)
    print("CLEANING UP", file=sys.stderr)
    shutil.rmtree(os.path.join(os.getcwd(), directory))
    return num

if __name__ == "__main__":
    potdata = ['ismailpromus/skin-diseases-image-dataset',"shakyadissanayake/oily-dry-and-normal-skin-types-dataset",
               "amellia/face-skin-disease",
               "andreibadescu10/skin-diseases-dataset-3",
               "syedalinaqvi/augmented-skin-conditions-image-dataset",
               "pacificrm/skindiseasedataset", 
               "hereisburak/pins-face-recognition"]
    keywords = ['eczema','rosacea',"acne","healthy", "clear", "oily", "dry", "sun", "normal", "pins"]
    preprocess(potdata, keywords, "train_data")
    
               


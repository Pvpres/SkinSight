import kaggle as kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile
import shutil
import sys
import cv2
from concurrent.futures import ProcessPoolExecutor
from logging_config import setup_logging, get_logger, PerformanceLogger, DataLogger
    
# Function to normalize data (resize to 128x128 and center on face)
def preprocess_image(image_path, target, filename):
    logger = get_logger("preprocessing")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return False

    # Convert to grayscale and detect face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        logger.debug(f"No face detected in image: {image_path}")
        return False
    
    x, y, w, h = faces[0]  # OpenCV rectangle format: (x, y, width, height)
    face_crop = image[y:y + h, x:x + w]
    
    # Resize the cropped face to 224x224
    resized_face = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
    output_path = os.path.join(target, filename)
    cv2.imwrite(output_path, resized_face)
    logger.debug(f"Successfully processed image: {image_path}")
    return True

# Function to process files in parallel
def process_files_parallel(filepaths, target):
    """
    Processes files in parallel using ProcessPoolExecutor.
    """
    logger = get_logger("preprocessing")
    perf_logger = PerformanceLogger(logger)
    
    logger.info(f"Processing {len(filepaths)} files in parallel")
    perf_logger.start_timer("parallel_processing")
    
    moved = 0
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [executor.submit(preprocess_image, filepath, target, filename)
                   for filepath, target, filename in filepaths]
        # Collect results as they complete
        for i, future in enumerate(futures):
            if future.result():
                moved += 1
            if (i + 1) % 100 == 0:  # Log progress every 100 files
                perf_logger.log_progress(i + 1, len(filepaths), "parallel_processing")
    
    perf_logger.end_timer("parallel_processing", f"- Successfully processed {moved}/{len(filepaths)} images")
    return moved

def sort_files(directory, keyword, folder, cap):
    """
    Sort files by keyword and process them with face detection.
    
    Args:
        directory: Directory to search for files
        keyword: Keyword to filter files by
        folder: Target folder for processed files
        cap: Maximum number of files to process
    
    Returns:
        Number of successfully processed files
    """
    logger = get_logger("preprocessing")
    perf_logger = PerformanceLogger(logger)
    
    logger.info(f"Sorting {keyword} images")
    target = os.path.join(os.getcwd(), folder, keyword)
    
    if os.path.exists(target):
        logger.warning(f"Target directory already exists: {target}")
        return 0
    
    # Create folder
    os.makedirs(target, exist_ok=True)
    logger.debug(f"Created target directory: {target}")
    
    # Collect filepaths that will be processed
    filepaths = []
    for root, dirs, files in os.walk(directory):
        if keyword in root.lower():
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                    filepath = os.path.join(root, filename)
                    filepaths.append((filepath, target, filename))
                if len(filepaths) >= cap:
                    logger.info(f"Cap reached for {keyword}: {cap} pictures will be processed")
                    break
        if len(filepaths) >= cap:
            break
    
    logger.info(f"Found {len(filepaths)} files to process for keyword: {keyword}")
    
    # Process files in parallel
    perf_logger.start_timer(f"sort_files_{keyword}")
    moved = process_files_parallel(filepaths, target)
    perf_logger.end_timer(f"sort_files_{keyword}", f"- Successfully processed {moved} images")
    
    logger.info(f"Successfully processed {moved}/{len(filepaths)} {keyword} images")
    return moved

def download_and_filter(api, dataset, directory):
    """
    Download and extract a dataset from Kaggle.
    
    Args:
        api: Kaggle API instance
        dataset: Dataset name to download
        directory: Directory to store the dataset
    """
    logger = get_logger("preprocessing")
    perf_logger = PerformanceLogger(logger)
    
    # Create directory for dataset using name of dataset
    dataset_dir = os.path.join(directory, dataset.split("/")[0])
    
    # Check if dataset already exists
    if os.path.exists(dataset_dir):
        logger.info(f"Dataset already exists: {dataset}")
        return
    
    # Create directory
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info(f"Processing dataset: {dataset}")
    
    # Create zip path for dataset
    zip_path = os.path.join(dataset_dir, f"{dataset.split('/')[-1]}.zip")
    
    # Download dataset as zipfile for speed and storage efficiency
    perf_logger.start_timer(f"download_{dataset}")
    api.dataset_download_files(dataset, path=dataset_dir, unzip=False)
    perf_logger.end_timer(f"download_{dataset}")
    
    # Extract contents from zipfile
    logger.info(f"Extracting dataset: {dataset}")
    perf_logger.start_timer(f"extract_{dataset}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    perf_logger.end_timer(f"extract_{dataset}")
    
    # Remove zipfile
    os.remove(zip_path)
    logger.debug(f"Removed temporary zip file: {zip_path}")
    logger.info(f"Successfully processed dataset: {dataset}")


def preprocess(datasets, keywords, directory):
    """
    Main preprocessing function that downloads datasets and processes images.
    
    Args:
        datasets: List of Kaggle dataset names to download
        keywords: List of keywords to filter images by
        directory: Directory to store downloaded datasets
    
    Returns:
        Total number of successfully processed images
    """
    logger = get_logger("preprocessing")
    data_logger = DataLogger(logger)
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting preprocessing pipeline")
    perf_logger.start_timer("total_preprocessing")
    
    # Initialize Kaggle API
    logger.info("Initializing Kaggle API")
    api = KaggleApi()
    api.authenticate()
    logger.info("Kaggle API authenticated successfully")
    
    # Download and filter datasets
    logger.info(f"Processing {len(datasets)} datasets")
    for dataset in datasets:
        download_and_filter(api, dataset, directory)
    
    # Process images by keyword
    logger.info(f"Processing images for {len(keywords)} keywords")
    total_processed = 0
    for keyword in keywords:
        processed_count = sort_files(directory, keyword, "usable_data", 6500)
        total_processed += processed_count
        logger.info(f"Keyword '{keyword}': {processed_count} images processed")
    
    # Rename pins folder to healthy
    pins_path = os.path.join(os.getcwd(), "usable_data/pins")
    healthy_path = os.path.join(os.getcwd(), "usable_data/healthy")
    if os.path.exists(pins_path):
        os.rename(pins_path, healthy_path)
        logger.info("Renamed 'pins' folder to 'healthy'")
    
    # Cleanup temporary files
    logger.info("Cleaning up temporary files")
    temp_dir = os.path.join(os.getcwd(), directory)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info(f"Removed temporary directory: {temp_dir}")
    
    perf_logger.end_timer("total_preprocessing", f"- Total processed: {total_processed} images")
    data_logger.log_data_quality(total_processed, total_processed, 0)  # Assuming no failures for now
    
    logger.info("Preprocessing pipeline completed successfully")
    return total_processed

if __name__ == "__main__":
    # Initialize logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting DermaHelper preprocessing pipeline")
    
    # Dataset configuration
    datasets = [
        'ismailpromus/skin-diseases-image-dataset',
        "shakyadissanayake/oily-dry-and-normal-skin-types-dataset",
        "amellia/face-skin-disease",
        "andreibadescu10/skin-diseases-dataset-3",
        "syedalinaqvi/augmented-skin-conditions-image-dataset",
        "pacificrm/skindiseasedataset", 
        "hereisburak/pins-face-recognition",
        "osmankagankurnaz/acne-dataset-in-yolov8-format"
    ]
    keywords = ['eczema', 'rosacea', "acne", "oily", "dry", "normal", "pins"]
    
    try:
        total_processed = preprocess(datasets, keywords, "train_data")
        logger.info(f"Preprocessing completed successfully. Total images processed: {total_processed}")
    except Exception as e:
        logger.error(f"Preprocessing failed with error: {str(e)}")
        raise
    
               


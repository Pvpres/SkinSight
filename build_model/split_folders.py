import splitfolders
import os
from logging_config import setup_logging, get_logger, PerformanceLogger

def split_data():
    """
    Split the processed data into train/validation/test sets.
    """
    logger = setup_logging(log_level="INFO")
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting data splitting process")
    
    # Define paths
    input_path = os.path.join(os.getcwd(), "usable_data")
    output_path = os.path.join(os.getcwd(), "usable_data")
    
    # Validate input path exists
    if not os.path.exists(input_path):
        logger.error(f"Input directory not found: {input_path}")
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    logger.info(f"Splitting data from {input_path} to {output_path}")
    logger.info("Split ratio: 80% train, 10% validation, 10% test")
    
    perf_logger.start_timer("data_splitting")
    
    try:
        # Split the data
        splitfolders.ratio(
            input_path, 
            output=output_path, 
            seed=42, 
            ratio=(.8, .1, .1), 
            move=True
        )
        perf_logger.end_timer("data_splitting", "- Data splitting completed successfully")
        
        # Clean up original folders
        logger.info("Cleaning up original folders")
        keywords = ["eczema", "rosacea", "acne", "healthy", "oily", "dry", "normal"]
        
        for keyword in keywords:
            folder_path = os.path.join(os.getcwd(), "usable_data", keyword)
            if os.path.exists(folder_path):
                try:
                    os.removedirs(folder_path)
                    logger.debug(f"Removed folder: {folder_path}")
                except OSError as e:
                    logger.warning(f"Could not remove folder {folder_path}: {e}")
        
        logger.info("Data splitting process completed successfully")
        
    except Exception as e:
        logger.error(f"Data splitting failed: {str(e)}")
        raise

if __name__ == "__main__":
    split_data()
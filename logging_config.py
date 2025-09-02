import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for the DermaHelper project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_file: Specific log file name (if None, auto-generated)
        console_output: Whether to output to console
        file_output: Whether to output to file
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if file_output:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"dermahelper_{timestamp}.log"
    
    log_file_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger("DermaHelper")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Error file handler for critical errors
    if file_output:
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "errors.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (if None, returns root DermaHelper logger)
    
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger("DermaHelper")
    return logging.getLogger(f"DermaHelper.{name}")


# Performance logging utilities
class PerformanceLogger:
    """Utility class for logging performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
        self.logger.info(f"Starting {operation}")
    
    def end_timer(self, operation: str, additional_info: str = ""):
        """End timing an operation and log the duration."""
        if operation in self.start_times:
            duration = datetime.now() - self.start_times[operation]
            self.logger.info(f"Completed {operation} in {duration.total_seconds():.2f}s {additional_info}")
            del self.start_times[operation]
    
    def log_progress(self, current: int, total: int, operation: str):
        """Log progress for long-running operations."""
        percentage = (current / total) * 100
        self.logger.info(f"{operation}: {current}/{total} ({percentage:.1f}%)")


# Data validation logging
class DataLogger:
    """Utility class for logging data-related operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_dataset_info(self, dataset_path: str, class_counts: dict):
        """Log information about a dataset."""
        self.logger.info(f"Dataset loaded from: {dataset_path}")
        self.logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            self.logger.info(f"  {class_name}: {count} samples")
    
    def log_data_quality(self, total_images: int, processed_images: int, failed_images: int):
        """Log data quality metrics."""
        success_rate = (processed_images / total_images) * 100 if total_images > 0 else 0
        self.logger.info(f"Data processing complete:")
        self.logger.info(f"  Total images: {total_images}")
        self.logger.info(f"  Successfully processed: {processed_images}")
        self.logger.info(f"  Failed: {failed_images}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
    
    def log_model_metrics(self, epoch: int, train_loss: float, val_loss: float, accuracy: float):
        """Log model training metrics."""
        self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")


# Initialize default logging
if __name__ == "__main__":
    # Example usage
    logger = setup_logging(log_level="DEBUG")
    logger.info("Logging system initialized")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message") 
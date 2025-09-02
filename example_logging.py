#!/usr/bin/env python3
"""
Example script demonstrating the logging system for DermaHelper.

This script shows how to:
1. Set up logging with different configurations
2. Use the performance logging utilities
3. Use the data logging utilities
4. Handle errors properly with logging
"""

import time
import random
from logging_config import setup_logging, get_logger, PerformanceLogger, DataLogger


def example_basic_logging():
    """Demonstrate basic logging functionality."""
    print("\n=== Basic Logging Example ===")
    
    # Set up logging
    logger = setup_logging(log_level="DEBUG")
    
    logger.info("Starting basic logging example")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Show different log levels
    logger.info("Logging different levels:")
    logger.debug("Debug: Detailed information for debugging")
    logger.info("Info: General information about program execution")
    logger.warning("Warning: Something unexpected happened")
    logger.error("Error: A more serious problem occurred")
    logger.critical("Critical: A critical error that may prevent the program from running")


def example_performance_logging():
    """Demonstrate performance logging utilities."""
    print("\n=== Performance Logging Example ===")
    
    logger = setup_logging(log_level="INFO")
    perf_logger = PerformanceLogger(logger)
    
    # Simulate a long-running operation
    logger.info("Starting performance monitoring example")
    
    perf_logger.start_timer("data_processing")
    
    # Simulate processing 1000 items
    total_items = 1000
    for i in range(total_items):
        # Simulate some work
        time.sleep(0.001)
        
        # Log progress every 100 items
        if (i + 1) % 100 == 0:
            perf_logger.log_progress(i + 1, total_items, "data_processing")
    
    perf_logger.end_timer("data_processing", "- Successfully processed all items")
    
    # Multiple operations
    operations = ["download", "preprocess", "train", "evaluate"]
    for op in operations:
        perf_logger.start_timer(op)
        time.sleep(random.uniform(0.1, 0.5))  # Simulate work
        perf_logger.end_timer(op)


def example_data_logging():
    """Demonstrate data logging utilities."""
    print("\n=== Data Logging Example ===")
    
    logger = setup_logging(log_level="INFO")
    data_logger = DataLogger(logger)
    
    # Simulate dataset information
    class_counts = {
        "eczema": 1500,
        "rosacea": 1200,
        "acne": 1800,
        "oily": 2000,
        "dry": 1600,
        "normal": 1400,
        "healthy": 2200
    }
    
    data_logger.log_dataset_info("/path/to/dataset", class_counts)
    
    # Simulate data quality metrics
    total_images = 10000
    processed_images = 9500
    failed_images = 500
    
    data_logger.log_data_quality(total_images, processed_images, failed_images)
    
    # Simulate model training metrics
    for epoch in range(1, 6):
        train_loss = 1.0 - (epoch * 0.15) + random.uniform(-0.05, 0.05)
        val_loss = 1.1 - (epoch * 0.12) + random.uniform(-0.08, 0.08)
        accuracy = 0.6 + (epoch * 0.08) + random.uniform(-0.02, 0.02)
        
        data_logger.log_model_metrics(epoch, train_loss, val_loss, accuracy)


def example_error_handling():
    """Demonstrate error handling with logging."""
    print("\n=== Error Handling Example ===")
    
    logger = setup_logging(log_level="INFO")
    
    logger.info("Starting error handling example")
    
    try:
        # Simulate an operation that might fail
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.error(f"Division by zero error: {e}")
        logger.debug("This error occurred in the example_error_handling function")
    
    try:
        # Simulate another potential error
        with open("nonexistent_file.txt", "r") as f:
            content = f.read()
    except FileNotFoundError as e:
        logger.warning(f"File not found: {e}")
        logger.info("Continuing with default values")
    
    logger.info("Error handling example completed")


def example_different_configurations():
    """Demonstrate different logging configurations."""
    print("\n=== Different Logging Configurations ===")
    
    # Development configuration (more verbose)
    print("Development logging (DEBUG level):")
    dev_logger = setup_logging(log_level="DEBUG", console_output=True, file_output=False)
    dev_logger.debug("This debug message is visible in development")
    dev_logger.info("Development logging is more verbose")
    
    # Production configuration (less verbose, file output)
    print("\nProduction logging (INFO level, file output):")
    prod_logger = setup_logging(log_level="INFO", console_output=False, file_output=True)
    prod_logger.debug("This debug message is NOT visible in production")
    prod_logger.info("Production logging focuses on important information")
    prod_logger.warning("Warnings are still logged in production")


def main():
    """Run all logging examples."""
    print("DermaHelper Logging System Examples")
    print("=" * 50)
    
    example_basic_logging()
    example_performance_logging()
    example_data_logging()
    example_error_handling()
    example_different_configurations()
    
    print("\n" + "=" * 50)
    print("All examples completed. Check the 'logs' directory for log files.")


if __name__ == "__main__":
    main() 
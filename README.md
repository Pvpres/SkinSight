# DermaHelper

A comprehensive dermatological image classification system with proper logging, error handling, and performance monitoring.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DermaHelper
   ```

2. **Run the installation script:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Or install manually:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

### Usage

1. **Test the logging system:**
   ```bash
   python example_logging.py
   ```

2. **Preprocess data:**
   ```bash
   python preprocessing.py
   ```

3. **Split data into train/val/test:**
   ```bash
   python split_folders.py
   ```

4. **Train the model:**
   ```bash
   python training.py
   ```

## 📁 Project Structure

```
DermaHelper/
├── preprocessing.py      # Data preprocessing pipeline
├── training.py          # Model training pipeline
├── split_folders.py     # Data splitting utilities
├── test.py             # MPS availability test
├── logging_config.py    # Comprehensive logging system
├── config.py           # Configuration management
├── requirements.txt    # Dependencies
├── setup.py           # Package setup
├── install.sh         # Installation script
├── example_logging.py # Logging examples
└── logs/              # Log files directory
```

## 🔧 Features

### Comprehensive Logging System
- **Structured logging** with different levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Performance monitoring** with timing utilities
- **Data quality metrics** logging
- **Rotating log files** with size limits
- **Separate error logs** for critical issues

### Data Pipeline
- **Kaggle dataset downloads** with progress tracking
- **Face detection** using OpenCV
- **Parallel processing** for efficiency
- **Data validation** and quality checks
- **Automatic cleanup** of temporary files

### Model Training
- **Transfer learning** with EfficientNet
- **Data augmentation** with class-specific strategies
- **Early stopping** and learning rate scheduling
- **Performance metrics** tracking
- **Model checkpointing**

## 📊 Logging Examples

### Basic Usage
```python
from logging_config import setup_logging, get_logger

# Set up logging
logger = setup_logging(log_level="INFO")
logger.info("Starting processing...")
logger.error("An error occurred")
```

### Performance Monitoring
```python
from logging_config import PerformanceLogger

perf_logger = PerformanceLogger(logger)
perf_logger.start_timer("operation")
# ... do work ...
perf_logger.end_timer("operation", "- Completed successfully")
```

### Data Quality Logging
```python
from logging_config import DataLogger

data_logger = DataLogger(logger)
data_logger.log_dataset_info("/path/to/data", class_counts)
data_logger.log_data_quality(total=10000, processed=9500, failed=500)
```

## 🛠️ Configuration

The project uses a centralized configuration system in `config.py`:

```python
from config import get_config

config = get_config()
config.preprocessing.MAX_IMAGES_PER_CLASS = 5000
config.training.NUM_EPOCHS = 30
```

## 📈 Performance

- **Parallel processing** for image preprocessing
- **GPU acceleration** with MPS (Apple Silicon) support
- **Memory-efficient** data loading
- **Progress tracking** for long-running operations

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Run `pip install -r requirements.txt`
2. **Kaggle API errors**: Set up credentials in `~/.kaggle/kaggle.json`
3. **Memory issues**: Reduce batch size in `config.py`
4. **CUDA/MPS errors**: Check GPU availability with `python test.py`

### Log Files

Check the `logs/` directory for:
- `dermahelper_YYYYMMDD_HHMMSS.log` - Main application logs
- `errors.log` - Critical error logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

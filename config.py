"""
Configuration file for DermaHelper project.
Centralizes all hardcoded parameters and settings.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Face detection parameters
    FACE_DETECTION_SCALE_FACTOR: float = 1.3
    FACE_DETECTION_MIN_NEIGHBORS: int = 5
    
    # Image processing parameters
    TARGET_IMAGE_SIZE: tuple = (224, 224)
    SUPPORTED_IMAGE_FORMATS: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    
    # Processing limits
    MAX_IMAGES_PER_CLASS: int = 6500
    
    # Parallel processing
    MAX_WORKERS: Optional[int] = None  # None = use all available cores
    
    # Dataset keywords
    KEYWORDS: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.KEYWORDS is None:
            self.KEYWORDS = ['eczema', 'rosacea', 'acne', 'oily', 'dry', 'normal', 'pins']


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model parameters
    NUM_CLASSES: int = 7
    BASE_MODEL: str = 'efficientnet_b0'
    DROPOUT_RATE: float = 0.4
    
    # Training parameters
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-4
    PATIENCE: int = 5
    
    # Data augmentation
    AUGMENTATION_PROBABILITIES: Optional[Dict[str, float]] = None
    
    # Class weights for augmentation
    CLASS_AUGMENTATION_WEIGHTS: Optional[Dict[int, str]] = None
    
    # Regularization
    LABEL_SMOOTHING: float = 0.1
    
    def __post_init__(self):
        if self.AUGMENTATION_PROBABILITIES is None:
            self.AUGMENTATION_PROBABILITIES = {
                'horizontal_flip': 0.4,
                'brightness_contrast': 0.8,
                'rotation': 0.8,
                'gaussian_blur': 0.7,
                'color_jitter': 0.7
            }
        
        if self.CLASS_AUGMENTATION_WEIGHTS is None:
            self.CLASS_AUGMENTATION_WEIGHTS = {
                0: "heavy",  # eczema
                1: "default", # rosacea
                2: "heavy",   # acne
                3: "default", # oily
                4: "default", # dry
                5: "default", # normal
                6: "heavy"    # healthy
            }


@dataclass
class DataConfig:
    """Configuration for data management."""
    
    # Data paths
    BASE_DATA_PATH: str = "usable_data"
    TRAIN_PATH: str = None
    VAL_PATH: str = None
    TEST_PATH: str = None
    
    # Split ratios
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    
    # Random seed
    RANDOM_SEED: int = 42
    
    def __post_init__(self):
        if self.TRAIN_PATH is None:
            self.TRAIN_PATH = os.path.join(self.BASE_DATA_PATH, "train")
        if self.VAL_PATH is None:
            self.VAL_PATH = os.path.join(self.BASE_DATA_PATH, "val")
        if self.TEST_PATH is None:
            self.TEST_PATH = os.path.join(self.BASE_DATA_PATH, "test")


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    CONSOLE_OUTPUT: bool = True
    FILE_OUTPUT: bool = True
    MAX_LOG_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT: int = 5


@dataclass
class ModelConfig:
    """Configuration for model saving and loading."""
    
    MODEL_SAVE_DIR: str = "models"
    BEST_MODEL_NAME: str = "best_model.pth"
    BEST_OPTIMIZER_NAME: str = "best_optimizer.pth"
    CHECKPOINT_INTERVAL: int = 5  # Save checkpoint every N epochs


@dataclass
class DermaHelperConfig:
    """Main configuration class for DermaHelper."""
    
    preprocessing: PreprocessingConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    logging: LoggingConfig = None
    model: ModelConfig = None
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.model is None:
            self.model = ModelConfig()


# Default configuration instance
config = DermaHelperConfig()


def get_config() -> DermaHelperConfig:
    """Get the default configuration."""
    return config


def update_config(**kwargs) -> DermaHelperConfig:
    """Update configuration with new values."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    return config


# Environment-specific configurations
def get_development_config() -> DermaHelperConfig:
    """Get configuration optimized for development."""
    config = DermaHelperConfig()
    config.logging.LOG_LEVEL = "DEBUG"
    config.training.NUM_EPOCHS = 5  # Shorter training for development
    config.preprocessing.MAX_IMAGES_PER_CLASS = 100  # Smaller dataset for testing
    return config


def get_production_config() -> DermaHelperConfig():
    """Get configuration optimized for production."""
    config = DermaHelperConfig()
    config.logging.LOG_LEVEL = "INFO"
    config.logging.FILE_OUTPUT = True
    config.logging.CONSOLE_OUTPUT = False
    return config 
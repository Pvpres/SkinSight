import torch
from logging_config import setup_logging, get_logger

def test_mps_availability():
    #Test MPS (Metal Performance Shaders) availability for Apple Silicon.
    logger = setup_logging(log_level="INFO")
    logger.info("Testing MPS backend availability")
    if not torch.backends.mps.is_available():
            logger.info("MPS backend is not ready for use please download")
            return False
    else:
        logger.info("MPS backend is available and ready for use")
        return True




if __name__ == "__main__":
    test_mps_availability()
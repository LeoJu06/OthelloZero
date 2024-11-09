
import logging

def setup_logger(name: str, log_file : str, level=logging.DEBUG):

    """
    This function is the blueprint for a logger.
    Each logger is given a name as an attribute and
    the path to the file to which it has to write.
    """

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
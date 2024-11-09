
import logging

def setup_logger(name: str, log_file : str, level=logging.DEBUG):

    """
    This function sets up and configures a logger with a specified name.
    The logger writes log messages to a specified file and can be used to log
    information at various severity levels (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Parameters:
    - name (str): The name of the logger instance. This helps in identifying log entries.
    - log_file (str): The file path where the log messages will be written.
    - level (int, optional): The logging level that determines the minimum severity of
    messages that will be logged (default is DEBUG).

    Returns:
    - logger (logging.Logger): A configured logger instance that writes to the specified file.

    The logger uses a standardized format for log messages, which includes:
    - Timestamp of the log entry
    - Name of the logger
    - Severity level of the log message (e.g., DEBUG, INFO)
    - The actual log message

    """

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
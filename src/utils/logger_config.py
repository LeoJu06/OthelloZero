import os
from src.utils.setup_logger import setup_logger

LOGGER_DISABLED = {"logger_test": False}


def logger_config(name: str, log_filename: str):
    """Configures the logger.

    Parameters:
        name (str): The name of the logger.
        log_filename (str): The name of the log file where logs will be written.

    Returns:
        Logger object configured to write to the specified file.

    Raises:
        FileNotFoundError: If the 'logs' directory does not exist in the root directory.
    """

    # Get the path to the root directory (e.g., OthelloZero).
    base_dir = os.getcwd()

    # Join the path to the 'logs' directory.
    path_to_logs_dir = os.path.join(base_dir, "logs")

    # Ensure that the 'logs' directory exists; if not, raise an error.
    if not os.path.exists(path_to_logs_dir):
        raise FileNotFoundError(
            f""" 
        Directory '{path_to_logs_dir}' not found.
        Please make sure the program is executed from the root directory (OthelloZero), 
        so that all relative paths can be resolved correctly."""
        )

    # Combine the 'logs' directory with the log filename.
    log_file_path = os.path.join(path_to_logs_dir, log_filename)

    # Return the configured logger.
    return setup_logger(name, log_file=log_file_path)


logger_board = logger_config("Logger_board", "logger_board.log")
logger_mcts = logger_config("Logger_MCTS", "logger_MCTS.log")

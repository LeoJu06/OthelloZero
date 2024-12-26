"""Test for logger_config.py"""

import os
from src.utils.logger_config import logger_config


def test_logger_config():
    # Get the path to the root directory (e.g., OthelloZero).
    base_dir = os.getcwd()

    # Join the path to the 'logs' directory.
    path_to_logs_dir = os.path.join(base_dir, "logs")

    # Ensure that the 'logs' directory exists; if not, raise an error.
    assert os.path.exists(
        path_to_logs_dir
    ), f""" 
        Directory '{path_to_logs_dir}' not found.
        Please make sure the program is executed from the root directory (OthelloZero), 
        so that all relative paths can be resolved correctly."""

    # Define the log filename.
    log_filename = "test_log.log"
    log_file_path = os.path.join(path_to_logs_dir, log_filename)

    # Configure the logger.
    test_logger = logger_config("test_logger", log_filename=log_filename)

    # Log a test message.
    test_message = "Log was written during a pytest test."
    test_logger.info(test_message)

    # Check if the log file exists.
    assert os.path.exists(log_file_path), f"Log file '{log_file_path}' was not created."

    # Read the log file to check if the expected log message is present.
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        log_contents = log_file.read()

    # Assert that the test message is in the log file.
    assert (
        test_message in log_contents
    ), f"Expected log message '{test_message}' not found in {log_file_path}."

    # Additional check: verify that the log file is not empty.
    assert len(log_contents) > 0, f"Log file {log_file_path} is empty."

    ###
    # "Test passed: Log file '{log_file_path}' was created and contains the expected log message.")
    ###

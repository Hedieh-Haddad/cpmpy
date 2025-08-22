# --- START OF NEW FILE csv_logger.py ---

import csv
import os

# This module will hold a single, global instance of the logger
_logger_instance = None


class PSACSVLogger:
    """
    A simple logger to record hyperparameter tuning rounds to a CSV file.
    """

    def __init__(self, filepath, param_keys):
        self.filepath = filepath
        self.param_keys = sorted(list(param_keys))  # Ensures consistent column order

        # Create the directory for the CSV file if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        self.file_handle = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file_handle)

        # Write the CSV Header
        header = ['round', 'stagnation_round', 'phase', 'runtime', 'objective', 'status'] + self.param_keys
        self.writer.writerow(header)
        self.file_handle.flush()

    def log_round(self, round_data):
        """Writes a single row of data to the CSV file."""
        params = round_data.get('params', {})
        row = [
                  round_data.get('round', ''),
                  round_data.get('stagnation_round', ''),
                  round_data.get('phase', ''),
                  round_data.get('runtime', ''),
                  round_data.get('objective', ''),
                  round_data.get('status', '')
              ] + [params.get(key, '') for key in self.param_keys]

        self.writer.writerow(row)
        self.file_handle.flush()

    def close(self):
        """Closes the file handle."""
        if self.file_handle:
            self.file_handle.close()


# --- Global functions to control the logger ---

def init_logger(filepath, param_keys):
    """Initializes the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PSACSVLogger(filepath, param_keys)


def log_round(round_data):
    """Logs a round if the logger has been initialized."""
    if _logger_instance is not None:
        _logger_instance.log_round(round_data)


def close_logger():
    """Closes the global logger if it exists."""
    global _logger_instance
    if _logger_instance is not None:
        _logger_instance.close()
        _logger_instance = None

# --- END OF NEW FILE csv_logger.py ---
# cpmpy/tools/psa/psa_csv_logger.py
import csv
import os
import sys
import threading
from datetime import datetime
import json

# Try to import numpy for type checking, but don't make it a hard requirement.
try:
    import numpy as np
    NUMPY_AVAILABLE_LOGGER = True
except ImportError:
    NUMPY_AVAILABLE_LOGGER = False


def ensure_serializable(obj):
    """
    A crucial helper function that recursively converts numpy data types
    (like np.int64 or np.bool_) into native Python types. This is necessary
    because the standard `json` library cannot serialize numpy types, which are
    often returned by optimization libraries like scikit-optimize.
    """
    if isinstance(obj, dict):
        # If the object is a dictionary, apply this function to each value.
        return {key: ensure_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        # If it's a list or tuple, apply this function to each element.
        return [ensure_serializable(element) for element in obj]
    # Check if numpy is available and if the object is a generic numpy scalar type.
    if NUMPY_AVAILABLE_LOGGER and isinstance(obj, np.generic):
        # .item() is the standard way to convert a numpy scalar to a native Python type.
        return obj.item()
    # If none of the above, return the object as is.
    return obj


class PSACSVLogger:
    """
    A singleton logger for the probing phase. It logs details of every trial
    to a detailed CSV file. Using the singleton pattern ensures that all parts of the
    code access the same logger instance without needing to pass it around.
    """
    _instance = None
    _lock = threading.Lock() # Ensures thread-safe file writes for multi-threaded applications

    def __new__(cls, *args, **kwargs):
        # The singleton pattern: only create a new instance if one doesn't already exist.
        if cls._instance is None:
            with cls._lock:
                # Double-check locking to prevent race conditions in multi-threaded contexts.
                if cls._instance is None:
                    cls._instance = super(PSACSVLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, filepath='psa_bo_log.csv', instance_name='unknown_instance', cli_args_str=''):
        # The __init__ only runs its logic once due to the hasattr check, preventing re-initialization.
        if not hasattr(self, '_initialized_logger_attributes'):
            self.filepath = filepath
            self.instance_name = instance_name
            self.cli_args_str = cli_args_str
            # Define the columns for the detailed probing log.
            self.fieldnames = [
                'timestamp', 'instance_name', 'cli_args',
                'probe_round', 'hyperparameters', 'timeout_used',
                'runtime_returned', 'objective_returned', 'status_returned',
                'is_best_so_far_runtime', 'is_best_so_far_objective', 'bo_objective_reported'
            ]
            self._initialize_file()
            self._initialized_logger_attributes = True

    def _initialize_file(self):
        """Creates the log file and writes the CSV header row if the file is new or empty."""
        file_exists_and_not_empty = False
        try:
            if os.path.isfile(self.filepath):
                if os.path.getsize(self.filepath) > 0:
                    file_exists_and_not_empty = True

            with open(self.filepath, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if not file_exists_and_not_empty:
                    writer.writeheader()
        except IOError as e:
            print(f"Error initializing CSV logger file {self.filepath}: {e}", file=sys.stderr)

    @classmethod
    def configure_instance(cls, filepath='psa_bo_log.csv', instance_name='unknown_instance', cli_args_str=''):
        """
        A class method to configure (or re-configure) the singleton logger instance.
        This is the primary way to set up the logger for a new run.
        """
        if cls._instance is None:
            # If no instance exists, create one using the provided arguments.
            cls._instance = cls(filepath, instance_name, cli_args_str)
        else:
            # If an instance already exists, just update its attributes for the new run.
            cls._instance.filepath = filepath
            cls._instance.instance_name = instance_name
            cls._instance.cli_args_str = cli_args_str
            cls._instance._initialize_file()
            cls._instance._initialized_logger_attributes = True
        return cls._instance

    @classmethod
    def get_instance(cls):
        """A standard way to access the singleton instance."""
        if cls._instance is None or not hasattr(cls._instance, '_initialized_logger_attributes'):
            # If accessed before configuration, configure with default values.
            return cls.configure_instance()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        Resets the singleton instance. This is useful in testing or when running
        multiple independent PSA runs within the same Python process.
        """
        with cls._lock:
            if cls._instance is not None and hasattr(cls._instance, '_initialized_logger_attributes'):
                delattr(cls._instance, '_initialized_logger_attributes')
            cls._instance = None

    def log_trial(self, probe_round, hyperparameters, timeout_used,
                  runtime_returned, objective_returned, status_returned,
                  is_best_so_far_runtime, is_best_so_far_objective, bo_objective_reported):
        """Logs a single row of data for one probing trial."""
        if not hasattr(self, '_initialized_logger_attributes'):
            print("Error: PSACSVLogger not initialized. Call configure_instance first.", file=sys.stderr)
            return

        # Sanitize hyperparameters for JSON serialization before writing.
        serializable_hyperparameters = ensure_serializable(hyperparameters) if hyperparameters else {}

        try:
            with self._lock: # Ensure only one thread writes at a time
                with open(self.filepath, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    row_data = {
                        'timestamp': datetime.now().isoformat(),
                        'instance_name': self.instance_name,
                        'cli_args': self.cli_args_str,
                        'probe_round': probe_round,
                        'hyperparameters': json.dumps(serializable_hyperparameters),
                        'timeout_used': timeout_used,
                        'runtime_returned': runtime_returned,
                        'objective_returned': objective_returned,
                        'status_returned': str(status_returned),
                        'is_best_so_far_runtime': is_best_so_far_runtime,
                        'is_best_so_far_objective': is_best_so_far_objective,
                        'bo_objective_reported': bo_objective_reported
                    }
                    writer.writerow(row_data)
        except IOError as e:
            print(f"Error writing to CSV log file {self.filepath}: {e}", file=sys.stderr)
        except TypeError as te:
            print(f"TypeError during JSON serialization in CSV logger: {te}", file=sys.stderr)
            print(f"Problematic hyperparameters data for json.dumps: {serializable_hyperparameters}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error in CSV logger log_trial: {e}", file=sys.stderr)


class PSASolvingPhaseLogger:
    """
    A separate, simpler logger for the final results of the entire PSA process.
    This logger overwrites previous results for the same instance/strategy combination,
    making it a clean summary file for easy comparison across different experiments.
    """
    _lock = threading.Lock()

    def __init__(self, filepath='solve_phase_results.csv'):
        self.filepath = filepath
        self.fieldnames = [
            'timestamp', 'instance_name', 'cli_args', 'hpo_strategy_name',
            'best_hyperparameters', 'solving_time_budget',
            'final_runtime', 'final_objective', 'final_status'
        ]
        self._initialize_file()

    def _initialize_file(self):
        """Ensures the output directory and the log file with its header exist."""
        with self._lock:
            output_dir = os.path.dirname(self.filepath)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            if not os.path.isfile(self.filepath) or os.path.getsize(self.filepath) == 0:
                try:
                    with open(self.filepath, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                        writer.writeheader()
                except IOError as e:
                    print(f"Error initializing solve phase CSV logger file {self.filepath}: {e}", file=sys.stderr)

    def log_solve(self, instance_name, cli_args_str, hpo_strategy_name,
                  best_hyperparameters, solving_time_budget,
                  final_runtime, final_objective, final_status):
        """
        Reads the existing summary file, removes any old entry for the current
        instance/strategy pair, and appends the new result.
        """
        with self._lock:
            # Prepare the new row of data to be logged.
            new_row = {
                'timestamp': datetime.now().isoformat(),
                'instance_name': instance_name,
                'cli_args': cli_args_str,
                'hpo_strategy_name': hpo_strategy_name,
                'best_hyperparameters': json.dumps(
                    ensure_serializable(best_hyperparameters) if best_hyperparameters else {}),
                'solving_time_budget': solving_time_budget,
                'final_runtime': final_runtime,
                'final_objective': final_objective,
                'final_status': str(final_status)
            }

            # Read all existing rows from the file.
            existing_rows = []
            try:
                if os.path.isfile(self.filepath) and os.path.getsize(self.filepath) > 0:
                    with open(self.filepath, 'r', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            # Keep a row only if it's NOT a match for the one we're about to log.
                            if not (row['instance_name'] == instance_name and
                                    row['hpo_strategy_name'] == hpo_strategy_name):
                                existing_rows.append(row)
            except (IOError, FileNotFoundError) as e:
                print(f"Could not read existing solve log file, will create new. Error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Unexpected error reading solve log {self.filepath}: {e}. Overwriting file.", file=sys.stderr)
                existing_rows = []

            # Add the new result to the list.
            existing_rows.append(new_row)

            # Write everything back to the file, overwriting it with the updated data.
            try:
                with open(self.filepath, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
            except IOError as e:
                print(f"Error writing to solve phase CSV log file {self.filepath}: {e}", file=sys.stderr)
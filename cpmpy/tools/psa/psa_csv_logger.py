# cpmpy/tools/psa/psa_csv_logger.py
import csv
import os
import sys
import threading
from datetime import datetime
import json

try:
    import numpy as np
    NUMPY_AVAILABLE_LOGGER = True
except ImportError:
    NUMPY_AVAILABLE_LOGGER = False


def ensure_serializable(obj):
    """
    Recursively iterates through a data structure and converts numpy
    scalar types to native Python types.
    """
    if isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_serializable(element) for element in obj]
    if NUMPY_AVAILABLE_LOGGER and isinstance(obj, np.generic):
        return obj.item()
    return obj


class PSACSVLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PSACSVLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, filepath='psa_bo_log.csv', instance_name='unknown_instance', cli_args_str=''):
        if not hasattr(self, '_initialized_logger_attributes'):
            self.filepath = filepath
            self.instance_name = instance_name
            self.cli_args_str = cli_args_str
            self.fieldnames = [
                'timestamp', 'instance_name', 'cli_args',
                'probe_round', 'hyperparameters', 'timeout_used',
                'runtime_returned', 'objective_returned', 'status_returned',
                'is_best_so_far_runtime', 'is_best_so_far_objective', 'bo_objective_reported'
            ]
            self._initialize_file()
            self._initialized_logger_attributes = True

    def _initialize_file(self):
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
        if cls._instance is None:
            cls._instance = cls(filepath, instance_name, cli_args_str)
        else:
            cls._instance.filepath = filepath
            cls._instance.instance_name = instance_name
            cls._instance.cli_args_str = cli_args_str
            cls._instance._initialize_file()
            cls._instance._initialized_logger_attributes = True
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None or not hasattr(cls._instance, '_initialized_logger_attributes'):
            return cls.configure_instance()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls._lock:
            if cls._instance is not None and hasattr(cls._instance, '_initialized_logger_attributes'):
                delattr(cls._instance, '_initialized_logger_attributes')
            cls._instance = None

    def log_trial(self, probe_round, hyperparameters, timeout_used,
                  runtime_returned, objective_returned, status_returned,
                  is_best_so_far_runtime, is_best_so_far_objective, bo_objective_reported):

        if not hasattr(self, '_initialized_logger_attributes'):
            print("Error: PSACSVLogger not initialized. Call configure_instance first.", file=sys.stderr)
            return

        serializable_hyperparameters = ensure_serializable(hyperparameters) if hyperparameters else {}

        try:
            with self._lock:
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

        with self._lock:
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

            existing_rows = []
            try:
                if os.path.isfile(self.filepath) and os.path.getsize(self.filepath) > 0:
                    with open(self.filepath, 'r', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if not (row['instance_name'] == instance_name and row[
                                'hpo_strategy_name'] == hpo_strategy_name):
                                existing_rows.append(row)
            except (IOError, FileNotFoundError) as e:
                print(f"Could not read existing solve log file, will create new. Error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Unexpected error reading solve log {self.filepath}: {e}. Overwriting file.", file=sys.stderr)
                existing_rows = []

            existing_rows.append(new_row)

            try:
                with open(self.filepath, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
            except IOError as e:
                print(f"Error writing to solve phase CSV log file {self.filepath}: {e}", file=sys.stderr)
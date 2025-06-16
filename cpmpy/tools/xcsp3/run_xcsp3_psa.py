# cpmpy/tools/xcsp3/run_xcsp3_psa.py

import argparse
import gc
import os
import sys
import json
from timeit import default_timer as timer

# This block ensures that the script can find the 'cpmpy' package
# when run directly from the command line. It finds the project root
# (e.g., 'cpmpy-myVersion') by going up three levels from the current file's location
# and adds this root directory to Python's system path. This allows
# absolute imports like `from cpmpy.tools...` to work reliably.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary external and internal libraries
from pycsp3.parser.xparser import ParserXCSP3, CallbackerXCSP3
from cpmpy import SolverLookup
from cpmpy.tools.probing_solving import PSAFactory
from cpmpy.tools.psa.enum import TimeType, TimeoutEvolution, StopCondition, RoundTimeType
from cpmpy.tools.xcsp3.parser_callbacks import CallbacksCPMPy
from cpmpy.tools.psa.psa_csv_logger import PSACSVLogger, PSASolvingPhaseLogger, ensure_serializable

try:
    import numpy as np

    NUMPY_AVAILABLE_MAIN = True
except ImportError:
    NUMPY_AVAILABLE_MAIN = False

if __name__ == "__main__":
    # Disable Python's garbage collector for more consistent timing of short-lived solver runs.
    gc.disable()

    # Fetch all available solvers that CPMpy knows about to populate the --solver choices.
    available_solvers = SolverLookup.solvernames()

    # --- Argument Parsing ---
    # Set up the command-line interface to configure the tuning run.
    parser = argparse.ArgumentParser(
        description="A python application for using `Probe and Solve Algorithm` (PSA) with XCSP3 files.")

    # Core arguments defining the problem, solver, and output.
    parser.add_argument("--input", help="The path to the input XCSP3 file", required=True, type=str)
    parser.add_argument("--solver", help="The solver to use (e.g., 'xcsp:ace@2.4')", required=True, type=str,
                        choices=available_solvers)
    parser.add_argument("--output-dir", help="Directory to store solution and log files", required=False, type=str,
                        default="results")
    parser.add_argument("--global-time-limit", help="The global time limit for the entire process in seconds",
                        required=False, type=int, default=1800)
    parser.add_argument("--max-tries", help="Maximum number of hyperparameter configurations to try during probing",
                        required=False, type=int, default=50)
    parser.add_argument("--tuning-file", help="A JSON file with the solver's tunable and default hyperparameters.",
                        required=True)

    # Strategy selection for the HPO method.
    parser.add_argument("--hpo-strategy", help="The hyperparameter optimization strategy to use",
                        choices=["bayesian", "hamming"], default="bayesian", type=str)

    # Arguments for configuring the time management strategies within PSA.
    parser.add_argument("--global-time-strategy", help="Strategy for splitting time between probing and solving",
                        required=False, type=TimeType, choices=list(TimeType), default=TimeType.PERCENT)
    parser.add_argument("--percent", help="Percentage of global time for probing (0-1)", required=False, type=float,
                        default=0.2)
    parser.add_argument("--round-time-strategy", help="Strategy for determining the initial round timeout",
                        choices=list(RoundTimeType), default=RoundTimeType.STATIC, type=RoundTimeType)
    parser.add_argument("--time-evolution", help="Strategy for how the round timeout evolves over time",
                        choices=list(TimeoutEvolution), default=TimeoutEvolution.STATIC, type=TimeoutEvolution)
    parser.add_argument("--stop-strategy", help="Primary condition for stopping the probing phase (informational)",
                        choices=list(StopCondition), default=StopCondition.TIMEOUT, type=StopCondition)

    args = parser.parse_args()

    # --- File and Directory Setup ---
    # Create output directories if they don't exist.
    os.makedirs(args.output_dir, exist_ok=True)
    probe_log_dir = os.path.join(args.output_dir, "probe")
    os.makedirs(probe_log_dir, exist_ok=True)
    instance_basename = os.path.basename(args.input)
    instance_name_clean = os.path.splitext(instance_basename)[0]

    # Automatically generate log file paths based on the instance name and chosen strategy.
    probe_log_filepath = os.path.join(probe_log_dir, f"{instance_name_clean}_{args.hpo_strategy}.csv")
    solve_log_filepath = os.path.join(args.output_dir, "solve_phase_results.csv")
    solution_filepath = os.path.join(args.output_dir, f"{instance_name_clean}_solution.txt")

    # --- Logger Initialization ---
    # Convert CLI args to a JSON string for easy logging in the CSV files.
    cli_args_dict = vars(args)
    serializable_cli_args_dict = {}
    for key, value in cli_args_dict.items():
        if isinstance(value, (TimeType, RoundTimeType, TimeoutEvolution, StopCondition)):
            serializable_cli_args_dict[key] = value.name  # Use the enum's name for readability
        else:
            serializable_cli_args_dict[key] = value
    cli_args_str_for_log = json.dumps(serializable_cli_args_dict)

    # Configure the singleton logger instance with the paths and info for this specific run.
    PSACSVLogger.reset_instance()
    csv_logger_instance = PSACSVLogger.configure_instance(filepath=probe_log_filepath, instance_name=instance_basename,
                                                          cli_args_str=cli_args_str_for_log)

    # --- XCSP3 Parsing ---
    operation_start_time = timer()
    print(f"Parsing XCSP3 instance: {args.input}", file=sys.stderr)
    xcsp_parser_instance = ParserXCSP3(args.input)
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True  # Ensure it doesn't hang on large instances
    callbacker = CallbackerXCSP3(xcsp_parser_instance, callbacks)

    try:
        callbacker.load_instance()
        print(f" XCSP3 instance parsing and loading time: {timer() - operation_start_time:.4f}s", flush=True,
              file=sys.stderr)
    except Exception as e:
        print(f"  Error parsing/loading XCSP3 instance: {e}", flush=True, file=sys.stderr)
        sys.exit(1)

    # Check if the parser successfully created a CPMpy model.
    cb = callbacker.cb
    if cb.cpm_model is None:
        print("Critical Error: CPMpy model was not created after parsing. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- PSA Execution ---
    # Use the factory to create the fully configured PSA object.
    # This is where the chosen HPO strategy and time management components are assembled.
    psa = PSAFactory.create_psa_from_cli(args, cb.cpm_model)

    print(
        f"Starting PSA tuning process with '{args.hpo_strategy}' strategy. Global time limit: {args.global_time_limit}s, Max Tries: {args.max_tries}",
        file=sys.stderr)
    best_params = psa.tune(time_limit=args.global_time_limit, max_tries=args.max_tries)

    # --- Final Reporting ---
    psa_total_script_time = timer() - operation_start_time
    print(f"\nPSA Tuning Finished. Total script time (post-argparse): {psa_total_script_time:.4f}s", file=sys.stderr)
    print("Best parameters found:", file=sys.stderr)
    # Use the `ensure_serializable` helper to handle numpy types before printing to console.
    serializable_best_params = ensure_serializable(best_params)
    print(json.dumps(serializable_best_params, indent=2), file=sys.stderr)

    print(f"\nDetailed PROBING log written to: {csv_logger_instance.filepath}", file=sys.stderr)

    # Log the final results to the summary CSV file.
    solve_logger = PSASolvingPhaseLogger(filepath=solve_log_filepath)
    solve_logger.log_solve(instance_name=instance_basename, cli_args_str=cli_args_str_for_log,
                           hpo_strategy_name=psa.hpo_strategy_name, best_hyperparameters=best_params,
                           solving_time_budget=psa.solving_time_budget, final_runtime=psa.final_runtime,
                           final_objective=psa.final_objective, final_status=psa.final_status)
    print(f"Final SOLVING summary appended to: {solve_logger.filepath}", file=sys.stderr)

    # Write a human-readable summary of the best solution to a text file.
    try:
        with open(solution_filepath, "w") as f:
            f.write(f"Solver: {args.solver}\n")
            f.write(f"Instance: {instance_basename}\n")
            f.write(f"Status: {psa.final_status}\n")
            f.write(f"Runtime: {psa.final_runtime}\n")
            f.write(f"Objective: {psa.final_objective}\n")
            f.write(f"Best Parameters: {json.dumps(serializable_best_params, indent=2)}\n")
        print(f"Final solution summary written to: {solution_filepath}", file=sys.stderr)
    except Exception as e:
        print(f"Could not write final solution file: {e}", file=sys.stderr)
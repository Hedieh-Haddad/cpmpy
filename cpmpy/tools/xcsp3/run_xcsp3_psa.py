# cpmpy/tools/xcsp3/run_xcsp3_psa.py

import argparse
import gc
import os
import sys
import json
from timeit import default_timer as timer

# This block ensures that the script can find the 'cpmpy' package
# when run directly from the command line.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    gc.disable()
    available_solvers = SolverLookup.solvernames()

    parser = argparse.ArgumentParser(
        description="A python application for using `Probe and Solve Algorithm` (PSA) with XCSP3 files.")

    parser.add_argument("--input", help="The path to the input XCSP3 file", required=True, type=str)
    parser.add_argument("--solver", help="The solver to use", required=True, type=str, choices=available_solvers)
    parser.add_argument("--output-dir", help="Directory to store solution and log files", required=False, type=str,
                        default="results")
    parser.add_argument("--global-time-limit", help="The global time limit for the solver in seconds", required=False,
                        type=int, default=1800)
    parser.add_argument("--max-tries", help="Maximum number of hyperparameter configurations to try during probing",
                        required=False, type=int, default=20)
    parser.add_argument("--tuning-file", required=True, help="A json file with the hyperparameters.")

    # Reverted to only two choices
    parser.add_argument("--hpo-strategy", help="The hyperparameter optimization strategy to use",
                        choices=["bayesian", "hamming"], default="bayesian", type=str)

    parser.add_argument("--global-time-strategy", help="Strategy for splitting time between probing and solving",
                        required=False, type=TimeType, choices=list(TimeType), default=TimeType.PERCENT)
    parser.add_argument("--percent", help="Percentage of global time for probing (0-1)", required=False, type=float,
                        default=0.2)
    parser.add_argument("--round-time-strategy", help="Strategy for initial round timeout", choices=list(RoundTimeType),
                        default=RoundTimeType.STATIC, type=RoundTimeType)
    parser.add_argument("--time-evolution", help="Strategy for how round timeout evolves",
                        choices=list(TimeoutEvolution), default=TimeoutEvolution.STATIC, type=TimeoutEvolution)
    parser.add_argument("--stop-strategy", choices=list(StopCondition),
                        help="Strategy for stopping probing (informational)", default=StopCondition.TIMEOUT,
                        type=StopCondition)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    probe_log_dir = os.path.join(args.output_dir, "probe")
    os.makedirs(probe_log_dir, exist_ok=True)
    instance_basename = os.path.basename(args.input)
    instance_name_clean = os.path.splitext(instance_basename)[0]
    probe_log_filepath = os.path.join(probe_log_dir, f"{instance_name_clean}_{args.hpo_strategy}.csv")
    solve_log_filepath = os.path.join(args.output_dir, "solve_phase_results.csv")
    solution_filepath = os.path.join(args.output_dir, f"{instance_name_clean}_solution.txt")
    cli_args_dict = vars(args)
    serializable_cli_args_dict = {}
    for key, value in cli_args_dict.items():
        if isinstance(value, (TimeType, RoundTimeType, TimeoutEvolution, StopCondition)):
            serializable_cli_args_dict[key] = value.name
        else:
            serializable_cli_args_dict[key] = value
    cli_args_str_for_log = json.dumps(serializable_cli_args_dict)
    PSACSVLogger.reset_instance()
    csv_logger_instance = PSACSVLogger.configure_instance(filepath=probe_log_filepath, instance_name=instance_basename,
                                                          cli_args_str=cli_args_str_for_log)
    operation_start_time = timer()
    xcsp_parser_instance = ParserXCSP3(args.input)
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(xcsp_parser_instance, callbacks)
    try:
        xcsp3_load_start_time = timer()
        callbacker.load_instance()
        xcsp3_load_end_time = timer()
        t_parse = xcsp3_load_end_time - xcsp3_load_start_time
        print(f" XCSP3 instance parsing and loading time: {t_parse:.4f}s", flush=True, file=sys.stderr)
    except Exception as e:
        print(f"  Error parsing/loading XCSP3 instance: {e}", flush=True, file=sys.stderr)
        sys.exit(1)
    cb = callbacker.cb
    if cb.cpm_model is None:
        print("Critical Error: CPMpy model was not created after parsing. Exiting.", file=sys.stderr)
        sys.exit(1)
    psa = PSAFactory.create_psa_from_cli(args, cb.cpm_model)
    print(
        f"Starting PSA tuning process with '{args.hpo_strategy}' strategy. Global time limit: {args.global_time_limit}s, Max Tries: {args.max_tries}",
        file=sys.stderr)
    best_params = psa.tune(time_limit=args.global_time_limit, max_tries=args.max_tries)
    psa_total_script_time = timer() - operation_start_time
    print(f"\nPSA Tuning Finished. Total script time (post-argparse): {psa_total_script_time:.4f}s", file=sys.stderr)
    print("Best parameters found:", file=sys.stderr)
    serializable_best_params = ensure_serializable(best_params)
    print(json.dumps(serializable_best_params, indent=2), file=sys.stderr)
    print(f"\nDetailed PROBING log written to: {csv_logger_instance.filepath}", file=sys.stderr)
    solve_logger = PSASolvingPhaseLogger(filepath=solve_log_filepath)
    solve_logger.log_solve(instance_name=instance_basename, cli_args_str=cli_args_str_for_log,
                           hpo_strategy_name=psa.hpo_strategy_name, best_hyperparameters=best_params,
                           solving_time_budget=psa.solving_time_budget, final_runtime=psa.final_runtime,
                           final_objective=psa.final_objective, final_status=psa.final_status)
    print(f"Final SOLVING summary appended to: {solve_logger.filepath}", file=sys.stderr)
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
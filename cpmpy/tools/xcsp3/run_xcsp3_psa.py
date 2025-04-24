import argparse
import gc
import sys
from timeit import default_timer as timer

from pycsp3.parser.xparser import ParserXCSP3, CallbackerXCSP3

from cpmpy import SolverLookup
from cpmpy.tools.probing_solving import PSAFactory
from cpmpy.tools.psa.enum import TimeType, TimeoutEvolution, StopCondition, RoundTimeType
from cpmpy.tools.xcsp3.parser_callbacks import CallbacksCPMPy

if __name__ == "__main__":
    gc.disable()
    available_solvers = SolverLookup.solvernames()
    parser = argparse.ArgumentParser(
        description="A python application for using `Probe and Solve Algorithm` (PSA) with XCSP3 files.")
    parser.add_argument("--input", help="The path to the input XCSP3 file", required=True, type=str)
    parser.add_argument("--solver", help="The solver to use", required=True, type=str,  choices=available_solvers,
                        default=available_solvers[0])
    parser.add_argument("--output", help="The path to the output csv", required=False, type=str, default="output.csv")
    parser.add_argument("--global-time-limit", help="The global time limit for the solver", required=False, type=int,
                        default=1800)
    parser.add_argument("--global-time-strategy",
                        help="The strategy used for splitting the time budget between the probing and the solving phase",
                        required=False, type=TimeType, choices=TimeType, default=TimeType.PERCENT)
    parser.add_argument("--percent",
                        help="Percentage of the global time limit to use for probing (0-1)",
                        required=False, type=float, default=0.2)
    parser.add_argument("--max-tries",
                        help="",
                        required=False, type=int, default=20)
    parser.add_argument("--round-time-strategy",
                        help="The strategy used for splitting the probing time budget between the configurations",
                        choices=RoundTimeType, default=RoundTimeType.STATIC, type=RoundTimeType)
    parser.add_argument("--time-evolution", help="The strategy used for evolving the timeout during probing",
                        choices=TimeoutEvolution, default=TimeoutEvolution.STATIC, type=TimeoutEvolution)
    parser.add_argument("--stop-strategy", choices=StopCondition,
                        help="The strategy used for stopping the probing phase", default=StopCondition.TIMEOUT,
                        type=StopCondition)

    args = parser.parse_args()
    start_time = timer()

    parser = ParserXCSP3(args.input)
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)


    try:
        start_time = timer()
        callbacker.load_instance()
        end_time = timer()
        t_parse = end_time - start_time
        print(f" Parsing time {t_parse}", flush=True, file=sys.stderr)
    except Exception as e:
        print(f"  Error parsing: {e}", flush=True, file=sys.stderr)

    cb = callbacker.cb
    psa = PSAFactory.create_psa_from_cli(args, cb.cpm_model)
    psa.tune(args.global_time_limit, args.max_tries)

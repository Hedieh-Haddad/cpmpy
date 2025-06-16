# cpmpy/tools/psa/hamming_tuner.py

import time
import numpy as np
from timeit import default_timer as timer

from cpmpy import SolverLookup
from cpmpy.solvers.utils import param_combinations
from cpmpy.solvers.solver_interface import ExitStatus as CPMpyExitStatus
from .log import log
from .hpo_strategies import SolverThread  # Re-use the watchdog thread
from .psa_csv_logger import PSACSVLogger, ensure_serializable


class HammingTuner:
    """
    A standalone parameter tuner using Hamming distance and adaptive capping.
    This does NOT use the PSA probe/solve split. It uses the entire time
    budget for its iterative tuning process.
    """

    def __init__(self, solver_name, cpm_model, all_params, defaults, xpath=None):
        self.name = self.__class__.__name__
        self._solver_name = solver_name
        self._cpm_model = cpm_model
        self._all_params = all_params
        self._defaults = defaults
        self._xpath = xpath

        # Internal state
        self.best_params = self._defaults.copy()
        self.best_runtime = float('inf')
        self.best_obj = None

        self.final_status = CPMpyExitStatus.UNKNOWN
        self.final_runtime = 0.0
        self.final_objective = None

        self._param_order = list(self._all_params.keys())
        self._best_config_np = self._params_to_np([self._defaults])[0]
        combos = list(param_combinations(self._all_params))
        self._combos_np = self._params_to_np(combos)
        np.random.shuffle(self._combos_np)

    def tune(self, time_limit: int, max_tries: int = 100):
        start_time = timer()
        log("Hamming Tuner: Running with default config to get base runtime...", "info")

        init_kwargs = {}
        if "xcsp" in self._solver_name:
            init_kwargs["xpath"] = self._xpath

        # Initial run with defaults
        s = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        solve_kwargs = {"time_limit": int(time_limit), **self._defaults}
        self._run_solver_with_watchdog(s, solve_kwargs)

        if s.status().exitstatus in (CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE):
            self.best_runtime = s.status().runtime
            self.best_obj = s.objective_value()
            log(f"  Base runtime: {self.best_runtime}s, Base Obj: {self.best_obj}", "info")
        else:
            log(f"  Default run failed. Using full time_limit ({time_limit}s) as initial best_runtime.", "warning")
            self.best_runtime = time_limit

        # Main tuning loop
        i = 0
        while len(self._combos_np) and i < max_tries:
            time_left = time_limit - (timer() - start_time)
            if time_left <= 1:  # Not enough time for another run
                break

            # Adaptive capping: timeout for this run is the best runtime so far
            current_timeout = min(self.best_runtime, time_left)

            # Select and run next configuration
            params_to_test, params_np = self._select_next_config()
            full_params_dict = self._defaults.copy()
            full_params_dict.update(params_to_test)

            solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
            solve_kwargs = {"time_limit": int(current_timeout), **full_params_dict}
            self._run_solver_with_watchdog(solver, solve_kwargs)

            # Update best if improved
            if solver.status().exitstatus in (
            CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE) and solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                self.best_params.update(params_to_test)
                self._best_config_np = params_np
                self.best_obj = solver.objective_value()
                log(f"  Hamming: New best runtime found ({self.best_runtime}s), updating surrogate and adaptive cap.",
                    "info")

            i += 1

        # Set final results for logging
        self.final_runtime = self.best_runtime
        self.final_objective = self.best_obj
        self.final_status = CPMpyExitStatus.OPTIMAL if self.best_obj is not None else CPMpyExitStatus.UNKNOWN

        log(f"Hamming Tuning Finished. Best runtime: {self.final_runtime}, Best Obj: {self.final_objective}", "info")
        return self.best_params

    def _select_next_config(self):
        scores = np.count_nonzero(self._combos_np != self._best_config_np, axis=1)
        best_score_idx = np.argmin(scores)
        params_np = self._combos_np[best_score_idx]
        self._combos_np = np.delete(self._combos_np, best_score_idx, axis=0)
        params_dict = {key: val for key, val in zip(self._param_order, params_np)}
        return params_dict, params_np

    def _run_solver_with_watchdog(self, solver, solve_kwargs):
        # (This is the same watchdog helper from hpo_strategies.py)
        solver_timeout = solve_kwargs.get("time_limit", 5.0)
        watchdog_timeout = int(solver_timeout) + 5
        solve_kwargs["time_limit"] = int(solver_timeout)
        solver_thread = SolverThread(solver, solve_kwargs)
        solver_thread.start()
        solver_thread.join(timeout=watchdog_timeout)
        if solver_thread.is_alive():
            log(f"WATCHDOG: Solver thread timed out after {watchdog_timeout}s.", "warning")
            solver.status().exitstatus = CPMpyExitStatus.UNKNOWN
            solver.status().runtime = solver_timeout
            solver._objective_value = None
        elif solver_thread.exception:
            log(f"WATCHDOG: Exception in solver thread: {solver_thread.exception}", "error")
            solver.status().exitstatus = CPMpyExitStatus.ERROR
            solver.status().runtime = solver_timeout
            solver._objective_value = None

    def _params_to_np(self, combos):
        arr = [[params[key] for key in self._param_order] for params in combos]
        return np.array(arr)
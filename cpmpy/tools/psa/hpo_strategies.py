# cpmpy/tools/psa/hpo_strategies.py

import sys
from abc import ABC, abstractmethod
import numpy as np
import threading
from skopt import Optimizer
from skopt.utils import dimensions_aslist, point_asdict

import cpmpy
from cpmpy import SolverLookup
from cpmpy.solvers.solver_interface import SolverInterface, ExitStatus as CPMpyExitStatus
from cpmpy.solvers.utils import param_combinations
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.time_strategies import RoundTimeStrategy, TimeoutEvolutionStrategy, TuningGlobalTimeoutStrategy
from cpmpy.tools.psa.psa_csv_logger import PSACSVLogger, ensure_serializable


# =================================================================
# Watchdog Timer Implementation
# =================================================================
class SolverThread(threading.Thread):
    """
    A helper class to run a solver in a separate thread.
    This is the core component of the watchdog timer pattern. It allows the main
    thread to monitor the solver's execution and kill it if it hangs.
    """

    def __init__(self, solver, solve_kwargs):
        super().__init__()
        self.solver = solver
        self.solve_kwargs = solve_kwargs
        self.exception = None

    def run(self):
        """
        This method is executed when the thread starts. It calls the solver's
        solve() method and catches any exceptions that might occur.
        """
        try:
            self.solver.solve(**self.solve_kwargs)
        except Exception as e:
            self.exception = e
            log(f"Exception in solver thread: {e}", "error")


# =================================================================
# Abstract Base Class for All HPO Strategies
# =================================================================
class HPOStrategy(ABC):
    """
    An abstract base class that defines the common interface for all
    Hyperparameter Optimization (HPO) strategies. It manages the best-found
    parameters, stats, and the overall interaction with the PSA framework.
    """

    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy: TuningGlobalTimeoutStrategy,
                 timeout_evolution_strategy: TimeoutEvolutionStrategy, all_configs, defaults, xpath=None,
                 transformers=lambda x: x):
        # Configuration provided at creation time
        self.name = self.__class__.__name__
        self._solver_name = solver_name
        self._cpm_model: cpmpy.Model = cpm_model
        self._max_tries = max_tries
        self._all_configs = all_configs
        self._defaults = defaults if defaults else {}
        self._xpath = xpath  # Path to the .xml file for XCSP3 solvers
        self._transformers = transformers  # For transforming param values before sending to solver

        # Strategy components for time management
        self._round_type_strategy = round_type_strategy
        self._global_time_splitting_strategy = global_time_splitting_strategy
        self._timeout_evolution_strategy = timeout_evolution_strategy

        # Internal state that changes during the run
        self._current_timeout = 0
        self._solver: SolverInterface | None = None
        self._solution_list = []

        # Tracking the best solution found so far
        self._best_params = defaults.copy() if defaults else {}
        self._best_runtime = float('inf')
        self._best_obj = None
        self._best_status = CPMpyExitStatus.UNKNOWN

        # Logging
        self._csv_logger_instance: PSACSVLogger | None = None

    @abstractmethod
    def initialize(self, time_limit=None, max_tries=None):
        """
        Initializes the strategy before the tuning loop starts.
        This includes setting up timers and running any initial configurations.
        The `_best_obj` is set to the worst possible value depending on the optimization goal.
        """
        if self._cpm_model.objective_ is not None:
            self._best_obj = float('inf') if self._cpm_model.objective_is_min else float('-inf')
        else:
            self._best_obj = None  # This is a satisfaction problem

    def _get_csv_logger(self) -> PSACSVLogger:
        """Lazily gets the singleton logger instance."""
        if self._csv_logger_instance is None:
            self._csv_logger_instance = PSACSVLogger.get_instance()
        return self._csv_logger_instance

    def probing_should_continue(self):
        """Checks if the probing phase should continue based on time and try limits."""
        return not self._global_time_splitting_strategy.probe_phase_must_finish(self._current_timeout)

    def probing_phase(self):
        """
        Runs a single probing trial.
        This involves getting a solver instance, running the specific HPO logic,
        logging the results, and updating the best-known solution.
        """
        init_kwargs = {}
        if "xcsp" in self._solver_name:
            init_kwargs["xpath"] = self._xpath
        self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)

        params_that_were_tested, hpo_objective_reported = self._internal_probing_phase(self._solver)

        # After the trial, log everything to the detailed CSV file.
        if params_that_were_tested is not None:
            current_status = self._solver.status().exitstatus
            current_runtime = self._solver.status().runtime
            # Only consider objective valid if the run was successful
            current_objective = self._solver.objective_value() if current_status in (
            CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE) else None

            is_new_best, _ = self._is_new_best(current_status, current_runtime, current_objective)

            # Merge tuned params with defaults for a complete log entry
            effective_params = self._defaults.copy()
            effective_params.update(params_that_were_tested)

            self._get_csv_logger().log_trial(
                probe_round=self._global_time_splitting_strategy.round_counter,
                hyperparameters=effective_params,
                timeout_used=self._current_timeout,
                runtime_returned=current_runtime,
                objective_returned=current_objective,
                status_returned=current_status,
                is_best_so_far_runtime=is_new_best,
                is_best_so_far_objective=is_new_best,
                bo_objective_reported=hpo_objective_reported
            )

        # Update internal state with the new best solution if applicable
        self._register_better_result_if_needed(self._solver, params_that_were_tested)
        self._global_time_splitting_strategy.update_probe_phase()

    @abstractmethod
    def _internal_probing_phase(self, solver: SolverInterface) -> tuple[dict | None, float | None] | None:
        """
        The core logic of the HPO strategy. Each subclass must implement this to define
        how it selects and runs one configuration.
        """
        pass

    def update_current_timeout(self):
        """Evolves the timeout for the next round based on the chosen time evolution strategy."""
        self._current_timeout = self._timeout_evolution_strategy.evolve(self._current_timeout, self._solution_list)
        time_left = self._global_time_splitting_strategy.probe_timeout - self._global_time_splitting_strategy.elapsed_time
        # Ensure the next timeout doesn't exceed the remaining probing time
        if self._current_timeout > time_left:
            self._current_timeout = max(0, time_left)
        log(f"Updated current_timeout for next probe: {self._current_timeout}s", "debug")

    def _get_solution_quality_score(self, status, objective, runtime):
        """
        Creates a comparable score tuple: (Status, Objective, -Runtime).
        This allows lexicographical comparison: OPTIMAL is best, then better objective, then faster runtime.
        """
        status_score = 0
        if status == CPMpyExitStatus.OPTIMAL:
            status_score = 2
        elif status == CPMpyExitStatus.FEASIBLE:
            status_score = 1

        obj_score = 0.0
        if objective is not None:
            obj_score = -objective if self._cpm_model.objective_is_min else objective

        runtime_score = -runtime if runtime is not None else -float('inf')

        return (status_score, obj_score, runtime_score)

    def _is_new_best(self, status, runtime, objective):
        """Compares a new result to the current best, returning True if it's better."""
        if status not in (CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE):
            return False, "Not a valid solution"

        current_score = self._get_solution_quality_score(self._best_status, self._best_obj, self._best_runtime)
        new_score = self._get_solution_quality_score(status, objective, runtime)

        return new_score > current_score, "New score is better" if new_score > current_score else "Current score is better or equal"

    def _register_better_result_if_needed(self, solver_that_ran, parameters_used_for_run):
        """If the solver's result is better than the current best, update the internal best-known stats."""
        if parameters_used_for_run is None: return

        is_better, reason = self._is_new_best(solver_that_ran.status().exitstatus, solver_that_ran.status().runtime,
                                              solver_that_ran.objective_value())

        if is_better:
            status, runtime, objective = solver_that_ran.status().exitstatus, solver_that_ran.status().runtime, solver_that_ran.objective_value()
            log(f"New best result found ({reason}). Obj: {objective}, Runtime: {runtime}s, Status: {status}", "info")
            self._best_obj, self._best_runtime, self._best_status = objective, runtime, status
            self._best_params = ensure_serializable(parameters_used_for_run.copy())

    def solving_phase(self):
        """
        The final phase where the best-found parameters are used for a final run with the remaining time.
        """
        final_params = self._defaults.copy()
        final_params.update(self._best_params)
        log(f"Solving phase: Using best tuned parameters: {final_params}", "info")

        final_params_as_strings = {k: str(self._transformers(v)) for k, v in final_params.items()}
        solving_timeout = self._global_time_splitting_strategy.solving_timeout

        init_kwargs = {}
        if "xcsp" in self._solver_name: init_kwargs["xpath"] = self._xpath
        self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)

        if solving_timeout is not None and round(solving_timeout) <= 0:
            log("Solving phase: No time left. Skipping final solve.", "warning")
            self._solver.status().runtime, self._solver.status().exitstatus, self._solver._objective_value = self._best_runtime, self._best_status, self._best_obj
            return

        self._run_solver_with_watchdog(self._solver, {"time_limit": round(solving_timeout), **final_params_as_strings})
        log(f"Solving phase finished. Status: {self._solver.status().exitstatus}, Runtime: {self._solver.status().runtime}, Obj: {self._solver.objective_value()}",
            "info")
        self._register_better_result_if_needed(self._solver, self._best_params)

    def finalize(self):
        """Returns the best hyperparameter configuration found."""
        return self._best_params.copy()

    def _params_to_np(self, combos):
        """Helper to convert a list of parameter dicts to a numpy array."""
        return np.array([[params.get(key) for key in self._param_order] for params in combos])

    def _np_to_params(self, arr):
        """Helper to convert a numpy array row back to a parameter dict."""
        return {key: val for key, val in zip(self._param_order, arr)}

    def _run_solver_with_watchdog(self, solver, solve_kwargs):
        """
        The watchdog timer implementation. It runs the solver in a separate thread and
        monitors it, preventing hangs.
        """
        solver_timeout = solve_kwargs.get("time_limit", 5.0)
        safe_solver_timeout = max(1, int(solver_timeout))
        watchdog_timeout = safe_solver_timeout + 5
        solve_kwargs["time_limit"] = safe_solver_timeout

        solver_thread = SolverThread(solver, solve_kwargs)
        solver_thread.start()
        solver_thread.join(timeout=watchdog_timeout)

        if solver_thread.is_alive():
            log(f"WATCHDOG: Solver thread hung and was terminated after {watchdog_timeout}s.", "warning")
            solver.status().exitstatus, solver.status().runtime, solver._objective_value = CPMpyExitStatus.UNKNOWN, safe_solver_timeout, None
            return False
        if solver_thread.exception:
            log(f"WATCHDOG: Exception in solver thread: {solver_thread.exception}", "error")
            solver.status().exitstatus, solver.status().runtime, solver._objective_value = CPMpyExitStatus.ERROR, safe_solver_timeout, None
            return False
        return True


# =================================================================
# Bayesian Optimization Strategy (with PSA)
# =================================================================
class BayesianOptimizationStrategy(HPOStrategy):
    """Uses scikit-optimize to perform Bayesian Optimization within the PSA framework."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opt = Optimizer(dimensions=dimensions_aslist(self._all_configs), base_estimator="GP", acq_func="EI")

    def initialize(self, time_limit=None, max_tries=None):
        super().initialize(time_limit, max_tries)
        self._global_time_splitting_strategy.update_global_timeout(time_limit)
        self._global_time_splitting_strategy.update_max_tries(max_tries)
        self._global_time_splitting_strategy.init()
        init_kwargs = {}
        if "xcsp" in self._solver_name: init_kwargs["xpath"] = self._xpath
        self._round_type_strategy.solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        self._round_type_strategy.init(self._global_time_splitting_strategy.probe_timeout)
        time_consumed = self._round_type_strategy.runtime
        self._global_time_splitting_strategy.update_probe_timeout(
            self._global_time_splitting_strategy.probe_timeout - time_consumed)
        self._current_timeout = self._round_type_strategy.round_timeout
        if self._current_timeout <= 0 and self._global_time_splitting_strategy.probe_timeout > 0:
            self._current_timeout = min(5, self._global_time_splitting_strategy.probe_timeout)
        elif self._current_timeout > self._global_time_splitting_strategy.probe_timeout:
            self._current_timeout = self._global_time_splitting_strategy.probe_timeout
        self._global_time_splitting_strategy.start_probe_phase()
        log(f"BO Strategy Initialized. Probe Budget: {self._global_time_splitting_strategy.probe_timeout}s", "info")

    def _internal_probing_phase(self, solver: SolverInterface) -> tuple[dict | None, float | None] | None:
        params_raw = self._opt.ask()
        params_dict = point_asdict(self._all_configs, params_raw)
        log(f"BO Probing (Round {self._global_time_splitting_strategy.round_counter}): ASK -> {params_dict}", "debug")

        final_params = self._defaults.copy()
        final_params.update(params_dict)
        final_params_as_strings = {k: str(self._transformers(v)) for k, v in final_params.items()}

        solve_timeout = max(0.1, self._current_timeout)
        self._run_solver_with_watchdog(solver, {"time_limit": solve_timeout, **final_params_as_strings})

        # Report a penalized objective to the optimizer for failed or timed-out runs.
        runtime = solver.status().runtime
        if solver.status().exitstatus not in (CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE):
            objective_to_report = solve_timeout * 1.5
        else:
            objective_to_report = runtime if runtime is not None else solve_timeout * 1.1

        self._opt.tell(params_raw, float(objective_to_report))
        return params_dict, objective_to_report


# =================================================================
# Hamming Distance Strategy (No PSA)
# =================================================================
class HammingDistanceNoPSAStrategy(HPOStrategy):
    """
    Uses Hamming Distance for local search and runs for the *entire* time budget.
    It does not use the PSA probe/solve split, making it a different kind of tuner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._param_order = list(self._all_configs.keys())
        self._best_config_np = None
        combos = list(param_combinations(self._all_configs))
        self._combos_np = self._params_to_np(combos)
        np.random.shuffle(self._combos_np)

    def initialize(self, time_limit=None, max_tries=None):
        super().initialize(time_limit, max_tries)
        self._global_time_splitting_strategy.update_global_timeout(time_limit)
        self._global_time_splitting_strategy.update_max_tries(max_tries)
        self._global_time_splitting_strategy.init()

        probe_timeout = self._global_time_splitting_strategy.probe_timeout
        log("Hamming (No PSA) Strategy: Running with default config to get base runtime...", "info")
        init_kwargs = {}
        if "xcsp" in self._solver_name: init_kwargs["xpath"] = self._xpath
        s = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        self._run_solver_with_watchdog(s, {"time_limit": probe_timeout, **self._defaults})

        self._register_better_result_if_needed(s, self._defaults)
        self._best_config_np = self._params_to_np([self._best_params])[0]
        self._current_timeout = self._best_runtime if self._best_status in (
        CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE) else probe_timeout
        self._global_time_splitting_strategy.start_probe_phase()

    def _get_next_hamming_config(self):
        if len(self._combos_np) == 0: return None
        if self._best_config_np is None: self._best_config_np = self._params_to_np([self._defaults])[0]
        scores = np.count_nonzero(self._combos_np != self._best_config_np, axis=1)
        idx = np.argmin(scores)
        params_np, params_dict = self._combos_np[idx], self._np_to_params(self._combos_np[idx])
        self._combos_np = np.delete(self._combos_np, idx, axis=0)
        return params_dict, params_np

    def _internal_probing_phase(self, solver: SolverInterface) -> tuple[dict | None, float | None] | None:
        result = self._get_next_hamming_config()
        if result is None:
            self._global_time_splitting_strategy._start_time = 0  # End probing
            return None, None

        params_dict, params_np = result
        log(f"Hamming NoPSA (Round {self._global_time_splitting_strategy.round_counter}): Closest -> {params_dict}",
            "debug")

        final_params = self._defaults.copy()
        final_params.update(params_dict)
        final_params_as_strings = {k: str(self._transformers(v)) for k, v in final_params.items()}

        time_left = self._global_time_splitting_strategy.probe_timeout - self._global_time_splitting_strategy.elapsed_time
        solve_timeout = min(self._current_timeout, time_left)

        self._run_solver_with_watchdog(solver, {"time_limit": solve_timeout, **final_params_as_strings})

        is_better, _ = self._is_new_best(solver.status().exitstatus, solver.status().runtime, solver.objective_value())
        if is_better:
            self._best_config_np = params_np
            # The adaptive cap is the runtime of the new best solution
            if solver.status().runtime: self._current_timeout = solver.status().runtime

        return params_dict, solver.status().runtime

    def solving_phase(self):
        """For NoPSA, this is just a final step to populate the solver object with the best results found."""
        log("Hamming (No PSA) Strategy: Finalizing results, no dedicated solving phase.", "info")
        if self._solver is None:
            init_kwargs = {}
            if "xcsp" in self._solver_name: init_kwargs["xpath"] = self._xpath
            self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)

        self._solver.status().runtime = self._best_runtime
        self._solver.status().exitstatus = self._best_status
        self._solver._objective_value = self._best_obj

    def update_current_timeout(self):
        # Timeout is handled adaptively within the probing phase for this strategy
        pass
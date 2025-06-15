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


class SolverThread(threading.Thread):
    def __init__(self, solver, solve_kwargs):
        super().__init__()
        self.solver = solver
        self.solve_kwargs = solve_kwargs
        self.result_status = None
        self.exception = None

    def run(self):
        try:
            self.solver.solve(**self.solve_kwargs)
        except Exception as e:
            self.exception = e
            log(f"Exception in solver thread: {e}", "error")


class HPOStrategy(ABC):
    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy: TuningGlobalTimeoutStrategy,
                 timeout_evolution_strategy: TimeoutEvolutionStrategy, all_configs, defaults, xpath=None,
                 transformers=lambda x: x):
        self.name = self.__class__.__name__
        self._solver_name = solver_name
        self._cpm_model: cpmpy.Model = cpm_model
        self._max_tries = max_tries
        self._round_type_strategy: RoundTimeStrategy = round_type_strategy
        self._global_time_splitting_strategy: TuningGlobalTimeoutStrategy = global_time_splitting_strategy
        self._timeout_evolution_strategy: TimeoutEvolutionStrategy = timeout_evolution_strategy
        self._current_timeout = 0
        self._best_params = defaults.copy() if defaults else {}
        self._best_runtime = float('inf')
        self._best_obj = None
        self._best_status = CPMpyExitStatus.UNKNOWN
        self._all_configs = all_configs
        self._defaults = defaults if defaults else {}
        self._solution_list = []
        self._seen_counter = 0
        self._transformers = transformers
        self._xpath = xpath
        self._solver: SolverInterface | None = None
        self._csv_logger_instance: PSACSVLogger | None = None

    @abstractmethod
    def initialize(self, time_limit=None, max_tries=None):
        if self._cpm_model.objective_ is not None:
            self._best_obj = float('inf') if self._cpm_model.objective_is_min else float('-inf')
        else:
            self._best_obj = None

    def _get_csv_logger(self) -> PSACSVLogger:
        if self._csv_logger_instance is None:
            self._csv_logger_instance = PSACSVLogger.get_instance()
        return self._csv_logger_instance

    def probing_should_continue(self):
        return not self._global_time_splitting_strategy.probe_phase_must_finish(self._current_timeout)

    def probing_phase(self):
        init_kwargs = dict()
        if "xcsp" in self._solver_name:
            init_kwargs["xpath"] = self._xpath
        self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        result_tuple = self._internal_probing_phase(self._solver)
        params_that_were_tested, hpo_objective_reported = (None, None)
        if result_tuple:
            params_that_were_tested, hpo_objective_reported = result_tuple
        effective_params_for_log = self._defaults.copy()
        if params_that_were_tested:
            effective_params_for_log.update(params_that_were_tested)
        if params_that_were_tested is not None:
            current_solver_status = self._solver.status().exitstatus
            current_solver_runtime = self._solver.status().runtime
            current_solver_objective = self._solver.objective_value()
            if current_solver_status not in (CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE):
                current_solver_objective = None
            is_new_best, _ = self._is_new_best(current_solver_status, current_solver_runtime, current_solver_objective)
            self._get_csv_logger().log_trial(
                probe_round=self._global_time_splitting_strategy.round_counter,
                hyperparameters=effective_params_for_log,
                timeout_used=self._current_timeout,
                runtime_returned=current_solver_runtime,
                objective_returned=current_solver_objective,
                status_returned=current_solver_status,
                is_best_so_far_runtime=is_new_best,
                is_best_so_far_objective=is_new_best,
                bo_objective_reported=hpo_objective_reported
            )
        self._register_better_result_if_needed(self._solver, params_that_were_tested)
        if params_that_were_tested is not None:
            self._register_solution(self._solver, params_that_were_tested)
        self._global_time_splitting_strategy.update_probe_phase()

    @abstractmethod
    def _internal_probing_phase(self, solver: SolverInterface) -> tuple[dict | None, float | None] | None:
        pass

    def update_current_timeout(self):
        self._current_timeout = self._timeout_evolution_strategy.evolve(self._current_timeout, self._solution_list)
        time_left = self._global_time_splitting_strategy.probe_timeout - self._global_time_splitting_strategy.elapsed_time
        if self._current_timeout > time_left:
            self._current_timeout = max(0, time_left)
        log(f"Updated current_timeout for next probe: {self._current_timeout}s", "debug")

    def _get_solution_quality_score(self, status, objective, runtime):
        status_score = 0
        if status == CPMpyExitStatus.OPTIMAL:
            status_score = 2
        elif status == CPMpyExitStatus.FEASIBLE:
            status_score = 1
        obj_score = 0.0
        if objective is not None:
            obj_score = -objective if self._cpm_model.objective_is_min else objective
        runtime_score = -runtime if runtime is not None else 0.0
        return (status_score, obj_score, runtime_score)

    def _is_new_best(self, status, runtime, objective):
        if status not in (CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE):
            return False, "Not a valid solution"
        current_score = self._get_solution_quality_score(self._best_status, self._best_obj, self._best_runtime)
        new_score = self._get_solution_quality_score(status, objective, runtime)
        if new_score > current_score:
            return True, "New score is better"
        return False, "Current score is better or equal"

    def _register_better_result_if_needed(self, solver_that_ran, parameters_used_for_run):
        if parameters_used_for_run is None: return
        status, runtime, objective = solver_that_ran.status().exitstatus, solver_that_ran.status().runtime, solver_that_ran.objective_value()
        is_better, reason = self._is_new_best(status, runtime, objective)
        if is_better:
            log(f"New best result found ({reason}). Obj: {objective}, Runtime: {runtime}s, Status: {status}", "info")
            self._best_obj, self._best_runtime, self._best_status, self._best_params = objective, runtime, status, ensure_serializable(
                parameters_used_for_run.copy())
            log(f"  Updated self._best_params to: {self._best_params}", "debug")

    def _register_solution(self, solver_that_ran, parameters_used_for_run):
        if parameters_used_for_run is None: return
        status = solver_that_ran.status().exitstatus
        if status in (CPMpyExitStatus.FEASIBLE, CPMpyExitStatus.OPTIMAL):
            objective = solver_that_ran.objective_value()
            if objective is not None:
                self._solution_list.append(
                    {'params': ensure_serializable(dict(parameters_used_for_run)), 'objective': objective,
                     'runtime': solver_that_ran.status().runtime, 'status': status})

    def solving_phase(self):
        final_params_config_for_solve = self._defaults.copy()
        if self._best_params:
            log(f"Solving phase: Using best tuned parameters found: {self._best_params}", "info")
            final_params_config_for_solve.update(self._best_params)
        else:
            log("Solving phase: No better tuned parameters found than defaults. Using full defaults.", "info")
        final_params_as_strings_for_solve = {k: str(self._transformers(v)) for k, v in
                                             final_params_config_for_solve.items()}
        solving_timeout_for_run = self._global_time_splitting_strategy.solving_timeout
        log(f"Starting solving phase with params: {final_params_as_strings_for_solve}, timeout: {solving_timeout_for_run}s",
            "info")
        init_kwargs = dict()
        if "xcsp" in self._solver_name:
            init_kwargs["xpath"] = self._xpath
        self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        if solving_timeout_for_run is not None and round(solving_timeout_for_run) <= 0:
            log("Solving phase: No time left. Skipping final solve.", "warning")
            return
        self._solver.solve(time_limit=round(solving_timeout_for_run), **final_params_as_strings_for_solve)
        log(f"Solving phase finished. Status: {self._solver.status().exitstatus}, Runtime: {self._solver.status().runtime}, Obj: {self._solver.objective_value()}",
            "info")
        self._register_better_result_if_needed(self._solver, ensure_serializable(final_params_config_for_solve))

    def finalize(self):
        return self._best_params.copy()

    def _params_to_np(self, combos):
        return np.array([[params.get(key) for key in self._param_order] for params in combos])

    def _np_to_params(self, arr):
        return {key: val for key, val in zip(self._param_order, arr)}

    def _run_solver_with_watchdog(self, solver, solve_kwargs):
        solver_timeout = solve_kwargs.get("time_limit", 5.0)
        safe_solver_timeout = max(1, int(solver_timeout))
        watchdog_timeout = safe_solver_timeout + 5
        solve_kwargs["time_limit"] = safe_solver_timeout
        solver_thread = SolverThread(solver, solve_kwargs)
        solver_thread.start()
        solver_thread.join(timeout=watchdog_timeout)
        if solver_thread.is_alive():
            log(f"WATCHDOG: Solver thread did not finish within {watchdog_timeout}s. Treating as a timeout.", "warning")
            solver.status().exitstatus, solver.status().runtime, solver._objective_value = CPMpyExitStatus.UNKNOWN, safe_solver_timeout, None
            return False
        if solver_thread.exception:
            log(f"WATCHDOG: An exception occurred in the solver thread: {solver_thread.exception}", "error")
            solver.status().exitstatus, solver.status().runtime, solver._objective_value = CPMpyExitStatus.ERROR, safe_solver_timeout, None
            return False
        return True


class BayesianOptimizationStrategy(HPOStrategy):
    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy, timeout_evolution_strategy: TimeoutEvolutionStrategy, all_configs,
                 defaults, xpath=None, transformers=lambda x: x):
        super().__init__(solver_name, cpm_model, max_tries, round_type_strategy,
                         global_time_splitting_strategy, timeout_evolution_strategy, all_configs, defaults,
                         xpath, transformers)
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
        log(f"BO Strategy Initialized. Initial CT: {self._current_timeout}s. Probe Budget: {self._global_time_splitting_strategy.probe_timeout}s",
            "info")

    def _internal_probing_phase(self, solver: SolverInterface) -> tuple[dict | None, float | None] | None:
        params_raw_from_ask = self._opt.ask()
        parameters_to_test_dict = point_asdict(self._all_configs, params_raw_from_ask)
        log(f"BO Probing (Round {self._global_time_splitting_strategy.round_counter}): ASK -> {parameters_to_test_dict}",
            "debug")
        final_params_for_solver = self._defaults.copy()
        final_params_for_solver.update(parameters_to_test_dict)
        final_params_as_strings = {k: str(self._transformers(v)) for k, v in final_params_for_solver.items()}
        solve_timeout = max(0.1, self._current_timeout)
        solve_kwargs = {"time_limit": solve_timeout, **final_params_as_strings}
        self._run_solver_with_watchdog(solver, solve_kwargs)
        if solver.status().exitstatus not in (
        CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE): solver._objective_value = None
        log(f"  Solver returned: Status={solver.status().exitstatus}, Runtime={solver.status().runtime}s, Obj={solver.objective_value()}",
            "debug")
        bo_objective_value_to_report = solver.status().runtime
        if solver.status().exitstatus not in (CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE):
            bo_objective_value_to_report = solve_timeout * 1.5
        elif solver.status().runtime is None or solver.status().runtime > solve_timeout:
            bo_objective_value_to_report = (solver.status().runtime or solve_timeout) * 1.1
        if isinstance(bo_objective_value_to_report,
                      np.generic): bo_objective_value_to_report = bo_objective_value_to_report.item()
        self._opt.tell(params_raw_from_ask, bo_objective_value_to_report)
        return parameters_to_test_dict, bo_objective_value_to_report


class HammingDistanceNoPSAStrategy(HPOStrategy):
    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy, timeout_evolution_strategy: TimeoutEvolutionStrategy, all_configs,
                 defaults, xpath=None, transformers=lambda x: x):
        super().__init__(solver_name, cpm_model, max_tries, round_type_strategy,
                         global_time_splitting_strategy, timeout_evolution_strategy, all_configs, defaults,
                         xpath, transformers)
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
        solve_kwargs = {"time_limit": probe_timeout, **self._defaults}
        self._run_solver_with_watchdog(s, solve_kwargs)
        self._register_better_result_if_needed(s, self._defaults)
        self._best_config_np = self._params_to_np([self._best_params])[0]
        self._current_timeout = self._best_runtime if self._best_status in (
        CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE) else probe_timeout
        self._global_time_splitting_strategy.start_probe_phase()

    def _get_score(self, combos_np):
        return np.count_nonzero(combos_np != self._best_config_np, axis=1)

    def _get_next_hamming_config(self):
        if len(self._combos_np) == 0:
            return None
        if self._best_config_np is None:
            self._best_config_np = self._params_to_np([self._defaults])[0]
        scores = self._get_score(self._combos_np)
        best_score_idx = np.argmin(scores)
        params_np = self._combos_np[best_score_idx]
        self._combos_np = np.delete(self._combos_np, best_score_idx, axis=0)
        return self._np_to_params(params_np), params_np

    def _internal_probing_phase(self, solver: SolverInterface) -> tuple[dict | None, float | None] | None:
        result = self._get_next_hamming_config()
        if result is None:
            self._global_time_splitting_strategy._start_time = 0
            return None, None
        parameters_to_test_dict, params_np = result
        log(f"Hamming NoPSA (Round {self._global_time_splitting_strategy.round_counter}): Closest -> {parameters_to_test_dict}",
            "debug")
        final_params_for_solver = self._defaults.copy()
        final_params_for_solver.update(parameters_to_test_dict)
        final_params_as_strings = {k: str(self._transformers(v)) for k, v in final_params_for_solver.items()}
        time_left = self._global_time_splitting_strategy.probe_timeout - self._global_time_splitting_strategy.elapsed_time
        solve_timeout = min(self._current_timeout, time_left)
        solve_kwargs = {"time_limit": solve_timeout, **final_params_as_strings}
        self._run_solver_with_watchdog(solver, solve_kwargs)
        if solver.status().exitstatus not in (
        CPMpyExitStatus.OPTIMAL, CPMpyExitStatus.FEASIBLE): solver._objective_value = None
        log(f"  Solver returned: Status={solver.status().exitstatus}, Runtime={solver.status().runtime}s, Obj={solver.objective_value()}",
            "debug")
        is_better, _ = self._is_new_best(solver.status().exitstatus, solver.status().runtime, solver.objective_value())
        if is_better:
            self._best_config_np = params_np
            if solver.status().runtime: self._current_timeout = solver.status().runtime
        return parameters_to_test_dict, solver.status().runtime

    def solving_phase(self):
        log("Hamming (No PSA) Strategy: Finalizing results, no dedicated solving phase.", "info")
        if self._solver is None:
            init_kwargs = {}
            if "xcsp" in self._solver_name: init_kwargs["xpath"] = self._xpath
            self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        self._solver.status().runtime, self._solver.status().exitstatus, self._solver._objective_value = self._best_runtime, self._best_status, self._best_obj

    def update_current_timeout(self):
        pass
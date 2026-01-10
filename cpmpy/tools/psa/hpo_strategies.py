import os
import sys
from abc import ABC, abstractmethod
import random as pyrandom
import numpy as np
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL
from pycsp3 import solver
from skopt import Optimizer
from skopt.utils import dimensions_aslist, point_asdict
from xcsp.solver.solver import Solver

import cpmpy
from cpmpy import SolverLookup
from cpmpy.solvers.solver_interface import SolverInterface, ExitStatus
from cpmpy.solvers import param_combinations
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.time_strategies import RoundTimeStrategy, TimeoutEvolutionStrategy, TuningGlobalTimeoutStrategy
from . import csv_logger
from .enum import StopCondition


class HPOStrategy(ABC):
    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy: TuningGlobalTimeoutStrategy,
                 timeout_evolution_strategy: TimeoutEvolutionStrategy, all_configs, defaults, xpath=None,
                 transformers=lambda x: x, stop_condition=StopCondition.TIMEOUT, stagnation_limit=10,
                 seed: int | None = None):
        self._solver_name = solver_name
        self._cpm_model: cpmpy.Model = cpm_model
        self._max_tries = max_tries
        self._round_type_strategy: RoundTimeStrategy = round_type_strategy
        self._global_time_splitting_strategy: TuningGlobalTimeoutStrategy = global_time_splitting_strategy
        self._timeout_evolution_strategy: TimeoutEvolutionStrategy = timeout_evolution_strategy
        self._current_timeout = 0  # CT
        self._best_runtime = None
        self._all_configs = all_configs
        self._defaults = defaults
        self._best_params = defaults
        self._best_obj = None
        self._solution_list = []
        self._seen_counter = 0
        self._transformers = transformers
        self._xpath = xpath
        self._solver: SolverInterface | None = None
        self.stop_condition = stop_condition
        self.stagnation_limit = stagnation_limit
        self.stagnation_counter = 0
        self.round_counter = 0
        self._found_first_solution = False
        self._seed = seed if seed is not None else 0

        # Flag to indicate strategy has nothing left to try
        self.exhausted = False

        os.environ["PYTHONHASHSEED"] = str(self._seed)
        pyrandom.seed(self._seed)
        np.random.seed(self._seed)

    @abstractmethod
    def initialize(self, time_limit=None, max_tries=None):
        self._best_obj = float('inf') if self._cpm_model.objective_is_min else float('-inf')

    def probing_should_continue(self):
        if self.exhausted:
            log("Stopping probing: Strategy exhausted search space.", "info")
            return False

        if self._global_time_splitting_strategy.probe_phase_must_finish(self._current_timeout):
            return False

        if self.stop_condition == StopCondition.FIRST_SOLUTION and self._found_first_solution:
            log("Stopping early: First solution has been found.", "info")
            return False

        if self.stop_condition == StopCondition.STAGNATION and self.stagnation_counter >= self.stagnation_limit:
            log(f"Stopping early: Stagnation limit of {self.stagnation_limit} reached.", "info")
            return False

        return True

    def probing_phase(self):
        init_kwargs = dict()
        if "xcsp" in self._solver_name:
            init_kwargs["xpath"] = self._xpath
        self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        objective_before_round = self._best_obj

        parameters = self._internal_probing_phase(self._solver)

        if parameters is None:
            return

        self.round_counter += 1
        if self._solver.objective_value() is not None:
            self._register_better_result_if_needed(self._solver, parameters)

        if self._best_obj == objective_before_round:
            self.stagnation_counter += 1

        self._register_solution(self._solver, parameters)
        self._global_time_splitting_strategy.update_probe_phase()

        csv_logger.log_round({
            'round': self.round_counter,
            'stagnation_round': self.stagnation_counter,
            'phase': 'probing',
            'runtime': self._solver.status().runtime,
            'objective': self._solver.objective_value(),
            'status': self._solver.status().exitstatus,
            'params': parameters
        })

    @abstractmethod
    def _internal_probing_phase(self, solver: SolverInterface):
        pass

    def update_current_timeout(self):
        self._current_timeout = self._timeout_evolution_strategy.evolve(self._current_timeout, self._solution_list)
        time_left = self._global_time_splitting_strategy.probe_timeout - self._global_time_splitting_strategy.elapsed_time
        if self._current_timeout > time_left:
            self._current_timeout = time_left

    @abstractmethod
    def solving_phase(self):
        pass

    @abstractmethod
    def finalize(self):
        pass

    def _register_better_result_if_needed(self, solver, parameters):
        if solver.status().exitstatus == ExitStatus.FEASIBLE or solver.status().exitstatus == ExitStatus.OPTIMAL:
            have_best_obj = (
                solver.objective_value() < self._best_obj
                if self._cpm_model.objective_is_min
                else solver.objective_value() > self._best_obj)

            same_obj = self._best_obj == solver.objective_value()
            better_runtime = self._best_runtime is None or solver.status().runtime < self._best_runtime

            if have_best_obj or (same_obj and better_runtime):
                log(f"Better obj or better runtime : {solver.objective_value()} in {solver.status().runtime}", "debug")
                self._best_obj = solver.objective_value()
                self._best_params = parameters
                self._best_runtime = round(solver.status().runtime, 3)
                log("Better obj or better runtime so we reset the counter", "debug")
                self.stagnation_counter = 1
                self._global_time_splitting_strategy.reset_counter()

    def _register_solution(self, solver: SolverInterface, parameters):
        if solver.status().exitstatus == ExitStatus.FEASIBLE or solver.status().exitstatus == ExitStatus.OPTIMAL:
            if not self._found_first_solution:
                self._found_first_solution = True
            self._solution_list.append({
                'params': dict(parameters),
                'objective': self._best_obj,
                'runtime': solver.status().runtime,
                'status': solver.status().exitstatus
            })


class BayesianOptimizationStrategy(HPOStrategy):
    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy, timeout_evolution_strategy, all_configs,
                 defaults, xpath=None, transformers=lambda x: x, stop_condition=StopCondition.TIMEOUT,
                 stagnation_limit=10, seed: int | None = None):
        super().__init__(solver_name, cpm_model, max_tries, round_type_strategy,
                         global_time_splitting_strategy, timeout_evolution_strategy, all_configs, defaults,
                         xpath, transformers, stop_condition, stagnation_limit, seed=seed)
        self._opt = Optimizer(
            dimensions=dimensions_aslist(self._all_configs),
            base_estimator="GP",
            acq_func="EI",
            random_state=self._seed
        )

    def initialize(self, time_limit=None, max_tries=None):
        super().initialize(time_limit, max_tries)
        self._global_time_splitting_strategy.update_global_timeout(time_limit)
        self._global_time_splitting_strategy.update_max_tries(max_tries)
        self._global_time_splitting_strategy.init()

        self._round_type_strategy.init(
            self._global_time_splitting_strategy.probe_timeout)

        # --- CAPTURE ROUND 0 (INITIALIZATION) ---
        if hasattr(self._round_type_strategy, 'solver') and self._round_type_strategy.solver is not None:
            s = self._round_type_strategy.solver
            # Important: Save this solver instance so solving_phase doesn't crash if probing is skipped
            self._solver = s

            if s.status().runtime is not None and s.status().runtime > 0:
                log(f"Captured Round 0 from Init Strategy (Bayes). Runtime: {s.status().runtime}", "info")
                self._register_better_result_if_needed(s, self._defaults)
                self._register_solution(s, self._defaults)
                csv_logger.log_round({
                    'round': 0,
                    'stagnation_round': 0,
                    'phase': 'init',
                    'runtime': s.status().runtime,
                    'objective': s.objective_value(),
                    'status': s.status().exitstatus,
                    'params': self._defaults
                })
        # ----------------------------------------

        self._global_time_splitting_strategy.update_probe_timeout(
            self._global_time_splitting_strategy.probe_timeout - self._round_type_strategy.runtime)

        log(str(self._global_time_splitting_strategy.probe_timeout), "debug")

        self._current_timeout = self._round_type_strategy.round_timeout
        log(str(self._current_timeout), "debug")
        self._global_time_splitting_strategy.start_probe_phase()

    def _internal_probing_phase(self, solver: SolverInterface):
        params = []
        parameters = {}

        for _ in range(10):
            params = self._opt.ask()
            parameters = point_asdict(self._all_configs,
                                      params) if self._global_time_splitting_strategy.round_counter > 0 else self._defaults

            seen_entry = next((s for s in self._solution_list if s.get('params') == parameters), None)

            if seen_entry is None:
                break

            log("Bayesian: Parameters seen before. Updating optimizer and retrying.", "debug")
            self._seen_counter += 1
            prev_obj = seen_entry.get('objective')
            if prev_obj is None:
                prev_obj = 1e9 if self._cpm_model.objective_is_min else -1e9

            y = prev_obj if self._cpm_model.objective_is_min else -prev_obj
            self._opt.tell(params, y)

        parameters = {k: self._transformers(v) for k, v in parameters.items()}
        parameters["check"] = True

        parameters["seed"] = int(self._seed)
        self._best_params.setdefault("seed", int(self._seed))

        log(f"New probing phase {parameters}", "debug")
        solver.solve(time_limit=max(int(self._current_timeout), 2), **parameters)

        if self._global_time_splitting_strategy.round_counter > 0:
            val = solver.objective_value()
            if val is None:
                val = 1e9 if self._cpm_model.objective_is_min else -1e9
            y = val if self._cpm_model.objective_is_min else -val
            self._opt.tell(params, y)

        return parameters

    def solving_phase(self):
        self._global_time_splitting_strategy.update_solving_timeout()
        log(
            f"Starting solving phase with {self._best_params} and {self._global_time_splitting_strategy.solving_timeout} seconds",
            "Info")

        log("Best parameters is same as defaults ? " + str(self._best_params == self._defaults), "debug")

        # --- FIX: Ensure solver is initialized if probing didn't run ---
        if self._solver is None:
            init_kwargs = dict()
            if "xcsp" in self._solver_name:
                init_kwargs["xpath"] = self._xpath
            self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        # ---------------------------------------------------------------

        self._best_params["seed"] = int(self._seed)

        self._solver.solve(time_limit=max(2, int(self._global_time_splitting_strategy.solving_timeout)),
                           **self._best_params)
        csv_logger.log_round({
            'round': 'No Round',
            'stagnation_round': "No Stagnation Round",
            'phase': 'solving',
            'runtime': self._solver.status().runtime,
            'objective': self._solver.objective_value(),
            'status': self._solver.status().exitstatus,
            'params': self._best_params
        })
        self._best_params = dict(self._best_params)
        self._best_params.setdefault("seed", self._seed)

    def finalize(self):
        return self._best_params


class HammingStrategy(HPOStrategy):
    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy, timeout_evolution_strategy, all_configs,
                 defaults, xpath=None, transformers=lambda x: x, stop_condition=StopCondition.TIMEOUT,
                 stagnation_limit=10, seed: int | None = None):
        super().__init__(solver_name, cpm_model, max_tries, round_type_strategy,
                         global_time_splitting_strategy, timeout_evolution_strategy, all_configs, defaults,
                         xpath, transformers, stop_condition, stagnation_limit, seed=seed)

        self._param_order = list(self._all_configs.keys())
        combos = list(param_combinations(self._all_configs))
        self._combos_np = self._params_to_np(combos)
        np.random.shuffle(self._combos_np)

    def _params_to_np(self, combos):
        if not combos:
            return np.array([])
        arr = [[params[key] for params in combos] for key in self._param_order]
        return np.array(arr).T

    def _dict_to_np_row(self, param_dict):
        return np.array([param_dict[key] for key in self._param_order])

    def initialize(self, time_limit=None, max_tries=None):
        super().initialize(time_limit, max_tries)
        self._global_time_splitting_strategy.update_global_timeout(time_limit)
        self._global_time_splitting_strategy.update_max_tries(max_tries)
        self._global_time_splitting_strategy.init()

        self._round_type_strategy.init(
            self._global_time_splitting_strategy.probe_timeout)

        # --- CAPTURE ROUND 0 (INITIALIZATION) ---
        if hasattr(self._round_type_strategy, 'solver') and self._round_type_strategy.solver is not None:
            s = self._round_type_strategy.solver
            # Important: Save this solver instance so solving_phase doesn't crash
            self._solver = s

            if s.status().runtime is not None and s.status().runtime > 0:
                log(f"Captured Round 0 from Init Strategy (Hamming). Runtime: {s.status().runtime}", "info")
                self._register_better_result_if_needed(s, self._defaults)
                self._register_solution(s, self._defaults)
                csv_logger.log_round({
                    'round': 0,
                    'stagnation_round': 0,
                    'phase': 'init',
                    'runtime': s.status().runtime,
                    'objective': s.objective_value(),
                    'status': s.status().exitstatus,
                    'params': self._defaults
                })
        # ----------------------------------------

        self._global_time_splitting_strategy.update_probe_timeout(
            self._global_time_splitting_strategy.probe_timeout - self._round_type_strategy.runtime)

        self._current_timeout = self._round_type_strategy.round_timeout
        self._global_time_splitting_strategy.start_probe_phase()

    def _internal_probing_phase(self, solver: SolverInterface):
        defaults_seen = any([s.get('params') == self._defaults for s in self._solution_list])

        if self._global_time_splitting_strategy.round_counter == 0 and not defaults_seen:
            parameters = self._defaults
        else:
            parameters = None
            while True:
                if len(self._combos_np) == 0:
                    log("Hamming: Search space exhausted.", "info")
                    self.exhausted = True
                    return None

                best_config_np = self._dict_to_np_row(self._best_params)
                scores = np.count_nonzero(self._combos_np != best_config_np, axis=1)
                best_score_idx = np.argmin(scores)
                params_np = self._combos_np[best_score_idx]

                self._combos_np = np.delete(self._combos_np, best_score_idx, axis=0)

                parameters = {key: val for key, val in zip(self._param_order, params_np)}

                seen = any([s.get('params') == parameters for s in self._solution_list])
                if not seen:
                    break

                log("Hamming: Parameters seen before. Skipping.", "debug")
                self._seen_counter += 1

        parameters_run = {k: self._transformers(v) for k, v in parameters.items()}
        parameters_run["check"] = True
        parameters_run["seed"] = int(self._seed)
        self._best_params.setdefault("seed", int(self._seed))

        log(f"New probing phase (Hamming) {parameters_run}", "debug")
        solver.solve(time_limit=max(int(self._current_timeout), 2), **parameters_run)

        return parameters

    def solving_phase(self):
        self._global_time_splitting_strategy.update_solving_timeout()
        log(f"Starting solving phase with {self._best_params}", "Info")

        # --- FIX: Ensure solver is initialized if probing didn't run ---
        if self._solver is None:
            init_kwargs = dict()
            if "xcsp" in self._solver_name:
                init_kwargs["xpath"] = self._xpath
            self._solver = SolverLookup.get(self._solver_name, self._cpm_model, **init_kwargs)
        # ---------------------------------------------------------------

        self._best_params["seed"] = int(self._seed)

        self._solver.solve(time_limit=max(2, int(self._global_time_splitting_strategy.solving_timeout)),
                           **self._best_params)
        csv_logger.log_round({
            'round': 'No Round',
            'stagnation_round': "No Stagnation Round",
            'phase': 'solving',
            'runtime': self._solver.status().runtime,
            'objective': self._solver.objective_value(),
            'status': self._solver.status().exitstatus,
            'params': self._best_params
        })
        self._best_params = dict(self._best_params)
        self._best_params.setdefault("seed", self._seed)

    def finalize(self):
        return self._best_params
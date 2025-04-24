import sys
from abc import ABC, abstractmethod

from pycsp3 import solver
from skopt import Optimizer
from skopt.utils import dimensions_aslist, point_asdict

import cpmpy
from cpmpy import SolverLookup
from cpmpy.solvers.solver_interface import SolverInterface
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.time_strategies import RoundTimeStrategy, TimeoutEvolutionStrategy, TuningGlobalTimeoutStrategy


class HPOStrategy(ABC):
    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy: TuningGlobalTimeoutStrategy,
                 timeout_evolution_strategy: TimeoutEvolutionStrategy, all_configs, defaults, transformers=lambda x: x):
        self._solver_name = solver_name
        self._cpm_model: cpmpy.Model = cpm_model
        self._max_tries = max_tries
        self._round_type_strategy: RoundTimeStrategy = round_type_strategy
        self._global_time_splitting_strategy: TuningGlobalTimeoutStrategy = global_time_splitting_strategy
        self._timeout_evolution_strategy: TimeoutEvolutionStrategy = timeout_evolution_strategy
        self._current_timeout = 0  # CT
        self._best_params = None
        self._best_runtime = None
        self._all_configs = all_configs
        self._defaults = defaults
        self._best_obj = None
        self._solution_list = []
        self._seen_counter = 0
        self._transformers = transformers

    @abstractmethod
    def initialize(self, time_limit=None, max_tries=None):
        self._best_obj = float('inf') if self._cpm_model.objective_is_min else float('-inf')

    def probing_should_continue(self):
        """
        Check if the probing phase should continue based on the elapsed time and the probe timeout.
        """
        return self._global_time_splitting_strategy.probe_phase_must_finish(self._current_timeout)

    def probing_phase(self):
        solver = SolverLookup.get(self._solver_name, self._cpm_model)
        parameters = self._internal_probing_phase(solver)
        if solver.objective_value() is not None:
            self._register_better_result_if_needed(solver, parameters)
        self._register_solution(solver, parameters)
        self._global_time_splitting_strategy.update_probe_phase()

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
        have_best_obj = solver.objective_value() < self._best_obj if self._cpm_model.objective_is_min else solver.objective_value > self._best_obj
        same_obj = self._best_obj == solver.objective_value()
        better_runtime = self._best_runtime is None or solver.status().runtime < self._best_runtime

        if have_best_obj or (same_obj and better_runtime):
            log(f"Better obj or better runtime : {solver.objective_value()} in {solver.status().runtime}", "debug")
            self._best_obj = solver.objective_value()
            self._best_params = parameters
            self._best_runtime = round(solver.status().runtime, 3)

    def _register_solution(self, solver: SolverInterface, parameters):
        self._solution_list.append({
            'params': dict(parameters),
            'objective': self._best_obj,
            'runtime': solver.status().runtime,
            'status': solver.status().exitstatus
        })


class BayesianOptimizationStrategy(HPOStrategy):
    """
    Bayesian Optimization strategy for hyperparameter tuning.
    """

    def __init__(self, solver_name, cpm_model, max_tries, round_type_strategy: RoundTimeStrategy,
                 global_time_splitting_strategy, timeout_evolution_strategy: TimeoutEvolutionStrategy, all_configs,
                 defaults):
        super().__init__(solver_name, cpm_model, max_tries, round_type_strategy,
                         global_time_splitting_strategy, timeout_evolution_strategy, all_configs, defaults)
        self._opt = Optimizer(dimensions=dimensions_aslist(self._all_configs), base_estimator="GP", acq_func="EI")

    def initialize(self, time_limit=None, max_tries=None):
        super().initialize(time_limit, max_tries)
        self._global_time_splitting_strategy.update_global_timeout(time_limit)
        self._global_time_splitting_strategy.update_max_tries(max_tries)
        self._global_time_splitting_strategy.init()

        self._round_type_strategy.init(
            self._global_time_splitting_strategy.probe_timeout)  # maybe launch a solver with default configuration and take `runtime` seconds.

        self._global_time_splitting_strategy.update_probe_timeout(
            self._global_time_splitting_strategy.probe_timeout - self._round_type_strategy.runtime)  # we update the PT by subtracting the time used by the runtime take by the first round (maybe 0 if the sub-strategy is Static for example)

        self._current_timeout = self._round_type_strategy.round_timeout
        self._global_time_splitting_strategy.start_probe_phase()

    def _internal_probing_phase(self, solver: SolverInterface):
        params = self._opt.ask()
        parameters = point_asdict(self._all_configs,
                                  params) if self._global_time_splitting_strategy.round_counter > 0 else self._defaults

        seen = any([s.get('param') == parameters for s in self._solution_list])

        if seen:
            log("Parameters seen before. Using stored results.", "info")
            self._seen_counter += 1
            return

        parameters = {k: self._transformers(v) for k, v in parameters.items()}
        solver.solve(time_limit=self._current_timeout, **parameters)
        return parameters

    def solving_phase(self):
        pass

    def finalize(self):
        return self._best_params, self._best_runtime

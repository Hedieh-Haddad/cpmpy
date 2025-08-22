import json
from argparse import Namespace

import numpy as np

import cpmpy
from cpmpy import SolverLookup
from cpmpy.solvers import param_combinations
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.tools import ParameterTuner
from cpmpy.tools.psa.enum import TimeoutEvolution, TimeType, RoundTimeType, HPOType, StopCondition
from cpmpy.tools.psa.hpo_strategies import HPOStrategy, BayesianOptimizationStrategy
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.time_strategies import TimeoutEvolutionStrategy, TuningGlobalTimeoutStrategy, RoundTimeStrategy, \
    TimeStrategyFactory
from timeit import default_timer as timer


class PSATuner(ParameterTuner):
    def __init__(self, hpo_strategy: HPOStrategy, solver_name: str, model: cpmpy.Model, all_params=None,
                 defaults=None):
        super().__init__(solver_name, model, all_params, defaults)
        self.hpo_strategy: HPOStrategy = hpo_strategy

    def tune(self, time_limit=None, max_tries=None, fix_params=None):
        if fix_params is None:
            fix_params = {}
        log(str(time_limit), "debug")
        log("Init hpo strategy...", "info")
        self.hpo_strategy.initialize(time_limit, max_tries)
        while self.hpo_strategy.probing_should_continue():
            log("Probing phase...", "info")
            self.hpo_strategy.probing_phase()
            log("Update timeout...", "info")
            self.hpo_strategy.update_current_timeout()
        self.hpo_strategy.solving_phase()
        return self.hpo_strategy.finalize()


class HammingTunner(ParameterTuner):
    def __init__(self, solver_name, cpm_model, all_params, defaults, xpath=None):
        super().__init__(solver_name, cpm_model, all_params, defaults)
        self._xpath = xpath
        self._defaults = defaults
        # Internal state
        self.best_params = self._defaults.copy()
        self.best_runtime = float('inf')
        self.best_obj = None

        self.final_status = ExitStatus.UNKNOWN
        self.final_runtime = 0.0
        self.final_objective = None

        self._best_config_np = self._params_to_np([self._defaults])[0]
        combos = list(param_combinations(self.all_params))
        self._combos_np = self._params_to_np(combos)
        np.random.shuffle(self._combos_np)

    def tune(self, time_limit=None, max_tries=None, fix_params=None):
        if fix_params is None:
            fix_params = {}
        start_time = timer()
        log("Hamming Tuner: Running with default config to get base runtime...", "info")

        init_kwargs = {}
        if "xcsp" in self.solvername:
            init_kwargs["xpath"] = self._xpath

        # Initial run with defaults
        s = SolverLookup.get(self.solvername, self.model, **init_kwargs)
        self._run_solver(s, self._defaults, time_limit=int(time_limit))

        if s.status().exitstatus in (ExitStatus.OPTIMAL, ExitStatus.FEASIBLE):
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

            solver = SolverLookup.get(self.solvername, self.model, **init_kwargs)
            self._run_solver(solver, full_params_dict, time_limit=int(current_timeout))

            # Update best if improved
            if solver.status().exitstatus in (
                    ExitStatus.OPTIMAL, ExitStatus.FEASIBLE) and solver.status().runtime < self.best_runtime:
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
        self.final_status = ExitStatus.OPTIMAL if self.best_obj is not None else ExitStatus.UNKNOWN

        log(f"Hamming Tuning Finished. Best runtime: {self.final_runtime}, Best Obj: {self.final_objective}", "info")
        return self.best_params

    def _select_next_config(self):
        scores = np.count_nonzero(self._combos_np != self._best_config_np, axis=1)
        best_score_idx = np.argmin(scores)
        params_np = self._combos_np[best_score_idx]
        self._combos_np = np.delete(self._combos_np, best_score_idx, axis=0)
        params_dict = {key: val for key, val in zip(self._param_order, params_np)}
        return params_dict, params_np

    def _run_solver(self, solver, solve_kwargs, time_limit=5):
        return solver.solve(time_limit=time_limit, **solve_kwargs)

    def _params_to_np(self, combos):
        arr = [[params[key] for key in self._param_order] for params in combos]
        return np.array(arr)


class TunerBuilder:
    def __init__(self, solver_name, cpm_model, xml_path=None):
        self._solver = solver_name
        self._cpm_model = cpm_model
        self._timeout_evolution_strategy: TimeoutEvolutionStrategy | None = None
        self._init_strategy: RoundTimeStrategy | None = None
        self._tuning_strategy: TuningGlobalTimeoutStrategy | None = None
        self._all_params = dict()
        self._defaults = dict()
        self._hpo_strategy = None
        self._xml_path = xml_path
        self._hamming = False
        self._stop_condition = StopCondition.TIMEOUT
        self._stagnation_limit = 10

    def build_timeout_evolution_strategy(self, timeout_evolution_type: TimeoutEvolution):
        self._timeout_evolution_strategy = TimeStrategyFactory.create_timeout_evolution_strategy(
            timeout_evolution_type)
        return self

    def build_round_time_splitting_strategy(self, round_time_splitting_strategy: RoundTimeType):
        self._init_strategy = TimeStrategyFactory.create_round_time_splitting_strategy(round_time_splitting_strategy,
                                                                                       SolverLookup.get(self._solver,
                                                                                                        self._cpm_model,
                                                                                                        xpath=self._xml_path),
                                                                                       self._defaults)
        return self

    def build_global_time_splitting_strategy(self, global_time_splitting_strategy: TimeType, global_timeout, percent):
        self._tuning_strategy = TimeStrategyFactory.create_global_time_splitting_strategy(
            global_time_splitting_strategy,
            global_timeout, percent)
        return self

    def build_hpo_strategy(self, hpo: HPOType):
        if hpo == HPOType.BAYESIAN_SEARCH:
            self._hpo_strategy = BayesianOptimizationStrategy(self._solver, self._cpm_model, 22, self._init_strategy,
                                                              self._tuning_strategy,
                                                              self._timeout_evolution_strategy,
                                                              self._all_params, self._defaults, xpath=self._xml_path,
                                                              stop_condition=self._stop_condition,
                                                              stagnation_limit=self._stagnation_limit)
        else:
            log(f"Unsupported HPO type: {hpo}", "error")
            raise ValueError(f"Unsupported HPO type: {hpo}")
        return self

    def build_tuning_parameters_from_file(self, tuning_file):
        with open(tuning_file) as f:
            parameters = json.load(f)
            self._all_params = parameters.get("tunable_params")
            self._defaults = parameters.get("default_params")
        return self

    def with_stop_condition(self, stop_condition: StopCondition, stagnation_limit: int):
        self._stop_condition = stop_condition
        self._stagnation_limit = stagnation_limit
        return self

    def build(self) -> PSATuner:
        return PSATuner(self._hpo_strategy, self._solver, self._cpm_model, self._all_params, self._defaults)


class PSAFactory:
    @staticmethod
    def create_psa_from_cli(args: Namespace, cpm_model) -> ParameterTuner:
        if args.hpo == HPOType.HAMMING_SEARCH:
            with open(args.tuning_file) as f:
                parameters = json.load(f)
                all_params = parameters.get("tunable_params")
                defaults = parameters.get("default_params")
                return HammingTunner(args.solver, cpm_model, all_params, defaults, args.input)

        builder = TunerBuilder(args.solver, cpm_model, args.input)
        if args.time_evolution != TimeoutEvolution.STATIC and args.round_time_strategy == RoundTimeType.FIRST_RUNTIME:
            log(f"First Runtime strategy is not compatible with dynamic timeout evolution", "error")
            raise ValueError("First Runtime strategy is not compatible with dynamic timeout evolution")

        builder.build_round_time_splitting_strategy(args.round_time_strategy).build_timeout_evolution_strategy(
            args.time_evolution).build_global_time_splitting_strategy(args.global_time_strategy,
                                                                      args.global_time_limit,
                                                                      args.percent).build_tuning_parameters_from_file(
            args.tuning_file).with_stop_condition(args.stop_strategy,
                                                  args.stagnation_limit).build_hpo_strategy(args.hpo)
        return builder.build()
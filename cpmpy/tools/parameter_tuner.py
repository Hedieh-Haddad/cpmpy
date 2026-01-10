import json
from argparse import Namespace

import numpy as np

import cpmpy
from cpmpy import SolverLookup
from cpmpy.solvers import param_combinations
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.tools import ParameterTuner
from cpmpy.tools.psa.enum import TimeoutEvolution, TimeType, RoundTimeType, HPOType, StopCondition
from cpmpy.tools.psa.hpo_strategies import HPOStrategy, BayesianOptimizationStrategy, HammingStrategy
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.time_strategies import TimeoutEvolutionStrategy, TuningGlobalTimeoutStrategy, RoundTimeStrategy, \
    TimeStrategyFactory
from timeit import default_timer as timer
from cpmpy.tools.psa import csv_logger


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
        self._stop_condition = StopCondition.TIMEOUT
        self._stagnation_limit = 10
        self._seed = 0

    def with_seed(self, seed: int):
        self._seed = seed
        return self

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
        # Common strategy arguments
        strategy_kwargs = {
            'solver_name': self._solver,
            'cpm_model': self._cpm_model,
            'max_tries': 1000,
            'round_type_strategy': self._init_strategy,
            'global_time_splitting_strategy': self._tuning_strategy,
            'timeout_evolution_strategy': self._timeout_evolution_strategy,
            'all_configs': self._all_params,
            'defaults': self._defaults,
            'xpath': self._xml_path,
            'stop_condition': self._stop_condition,
            'stagnation_limit': self._stagnation_limit,
            'seed': self._seed
        }

        if hpo == HPOType.BAYESIAN_SEARCH:
            self._hpo_strategy = BayesianOptimizationStrategy(**strategy_kwargs)
        elif hpo == HPOType.HAMMING_SEARCH:
            self._hpo_strategy = HammingStrategy(**strategy_kwargs)
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
        builder = TunerBuilder(args.solver, cpm_model, args.input)

        if args.time_evolution != TimeoutEvolution.STATIC and args.round_time_strategy == RoundTimeType.FIRST_RUNTIME:
            log(f"First Runtime strategy is not compatible with dynamic timeout evolution", "error")
            raise ValueError("First Runtime strategy is not compatible with dynamic timeout evolution")

        builder.build_round_time_splitting_strategy(args.round_time_strategy) \
            .build_timeout_evolution_strategy(args.time_evolution) \
            .build_global_time_splitting_strategy(args.global_time_strategy,
                                                  args.global_time_limit,
                                                  args.percent) \
            .build_tuning_parameters_from_file(args.tuning_file) \
            .with_stop_condition(args.stop_strategy, args.stagnation_limit) \
            .with_seed(getattr(args, "seed", 0)) \
            .build_hpo_strategy(args.hpo)

        return builder.build()
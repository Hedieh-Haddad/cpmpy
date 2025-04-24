from argparse import Namespace

import cpmpy
from cpmpy import SolverLookup
from cpmpy.tools import ParameterTuner
from cpmpy.tools.psa.enum import TimeoutEvolution, TimeType, RoundTimeType
from cpmpy.tools.psa.hpo_strategies import HPOStrategy
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.time_strategies import TimeoutEvolutionStrategy, TuningGlobalTimeoutStrategy, RoundTimeStrategy, \
    TimeStrategyFactory


class PSA(ParameterTuner):
    def __init__(self, hpo_strategy: HPOStrategy, solver_name: str, model: cpmpy.Model, all_params=None, defaults=None):
        super().__init__(solver_name, model, all_params, defaults)
        self.hpo_strategy: HPOStrategy = hpo_strategy

    def tune(self, time_limit=None, max_tries=None, fix_params=None):
        if fix_params is None:
            fix_params = {}
        self.hpo_strategy.initialize()
        while self.hpo_strategy.probing_should_continue():
            self.hpo_strategy.probing_phase()
            self.hpo_strategy.update_current_timeout()
        self.hpo_strategy.solving_phase()
        return self.hpo_strategy.finalize()


class PSABuilder:
    def __init__(self, solver_name, cpm_model):
        self._solver = solver_name
        self._cpm_model = cpm_model
        self._timeout_evolution_strategy: TimeoutEvolutionStrategy | None = None
        self._init_strategy: RoundTimeStrategy | None = None
        self._tuning_strategy: TuningGlobalTimeoutStrategy | None = None
        self._all_params = dict()
        self._defaults = dict()
        self._hpo_strategy = None

    def build_timeout_evolution_strategy(self, timeout_evolution_type: TimeoutEvolution):
        self._timeout_evolution_strategy = TimeStrategyFactory.create_timeout_evolution_strategy(timeout_evolution_type)
        return self

    def build_round_time_splitting_strategy(self, round_time_splitting_strategy: RoundTimeType):
        self._init_strategy = TimeStrategyFactory.create_round_time_splitting_strategy(round_time_splitting_strategy,
                                                                                       SolverLookup.get(self._solver,
                                                                                                        self._cpm_model),
                                                                                       self._defaults)
        return self

    def build_global_time_splitting_strategy(self, global_time_splitting_strategy: TimeType, global_timeout, percent):
        self._tuning_strategy = TimeStrategyFactory.create_global_time_splitting_strategy(
            global_time_splitting_strategy,
            global_timeout, percent)
        return self

    def _build_hpo_strategy(self):
        pass

    def build(self) -> PSA:
        self._build_hpo_strategy()
        return PSA(self._hpo_strategy, self._solver, self._cpm_model, self._all_params, self._defaults)


class PSAFactory:
    @staticmethod
    def create_psa_from_cli(args: Namespace, cpm_model) -> PSA:
        builder = PSABuilder(args.solver, cpm_model)

        if args.time_evolution != TimeoutEvolution.STATIC and args.round_time_strategy == RoundTimeType.FIRST_RUNTIME:
            log(f"First Runtime strategy is not compatible with dynamic timeout evolution","error")
            raise ValueError("First Runtime strategy is not compatible with dynamic timeout evolution")

        builder.build_round_time_splitting_strategy(args.round_time_strategy).build_timeout_evolution_strategy(
            args.time_evolution).build_global_time_splitting_strategy(args.global_timeout_strategy,
                                                                      args.global_time_limit, args.percent)
        return builder.build()

# cpmpy/tools/probing_solving.py

from cpmpy.tools.psa.hpo_strategies import HPOStrategy
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.enum import RoundTimeType, TimeoutEvolution
from cpmpy import Model


class PSA:
    def __init__(self, cpm_model: Model, hpo_strategy: HPOStrategy):
        self._cpm_model = cpm_model
        self._hpo_strategy = hpo_strategy
        self.final_solver = None

    def tune(self, time_limit: int, max_tries: int = 100):
        self._hpo_strategy.initialize(time_limit=time_limit, max_tries=max_tries)
        while self._hpo_strategy.probing_should_continue():
            self._hpo_strategy.probing_phase()
            self._hpo_strategy.update_current_timeout()
        self._hpo_strategy.solving_phase()
        self.final_solver = self._hpo_strategy._solver
        return self._hpo_strategy.finalize()

    @property
    def hpo_strategy_name(self):
        return self._hpo_strategy.name if self._hpo_strategy else "Unknown"

    @property
    def final_status(self):
        return self.final_solver.status().exitstatus if self.final_solver else "Not Run"

    @property
    def final_runtime(self):
        return self.final_solver.status().runtime if self.final_solver else 0.0

    @property
    def final_objective(self):
        return self.final_solver.objective_value() if self.final_solver else None

    @property
    def solving_time_budget(self):
        return self._hpo_strategy._global_time_splitting_strategy.solving_timeout


class PSABuilder:
    def __init__(self, cpm_model: Model):
        self._cpm_model = cpm_model
        self._hpo_strategy: HPOStrategy | None = None

    def with_hpo_strategy(self, hpo_strategy: HPOStrategy):
        self._hpo_strategy = hpo_strategy
        return self

    def build(self):
        return PSA(self._cpm_model, self._hpo_strategy)


class PSAFactory:
    @staticmethod
    def create_psa_from_cli(args, cpm_model):
        from cpmpy.tools.psa.hpo_strategies import BayesianOptimizationStrategy, HammingDistanceNoPSAStrategy
        from cpmpy.tools.psa.time_strategies import (
            PercentageTuningGlobalTimeoutStrategy,
            StaticRoundTimeStrategy, FirstRuntimeRoundTimeStrategy,
            StaticTimeoutEvolutionStrategy, DynamicGeometricTimeoutEvolutionStrategy,
            DynamicLubyTimeoutEvolutionStrategy
        )
        import json

        if args.tuning_file:
            with open(args.tuning_file, 'r') as f:
                tuning_config = json.load(f)
            tunable_params = tuning_config.get("tunable_params", {})
            default_params = tuning_config.get("default_params", {})
        else:
            log("Warning: No tuning file provided. Using empty hyperparameter space.", "warning")
            tunable_params, default_params = {}, {}

        if args.hpo_strategy == "hamming":
            log("Using 'hamming' (No PSA) strategy, forcing global time to 100% for tuning.", "info")
            time_split_percent = 1.0
        else:
            time_split_percent = args.percent

        global_time_strategy = PercentageTuningGlobalTimeoutStrategy(
            global_timeout=args.global_time_limit,
            percent=time_split_percent
        )

        if args.round_time_strategy == RoundTimeType.STATIC:
            round_time_strategy = StaticRoundTimeStrategy(solver=None, default_config=None)
        else:
            round_time_strategy = FirstRuntimeRoundTimeStrategy(solver=None, default_config=default_params)

        if args.time_evolution == TimeoutEvolution.STATIC:
            timeout_evolution_strategy = StaticTimeoutEvolutionStrategy()
        elif args.time_evolution == TimeoutEvolution.GEOMETRIC:
            timeout_evolution_strategy = DynamicGeometricTimeoutEvolutionStrategy()
        else:
            timeout_evolution_strategy = DynamicLubyTimeoutEvolutionStrategy()

        strategy_map = {
            "bayesian": BayesianOptimizationStrategy,
            "hamming": HammingDistanceNoPSAStrategy,
        }

        strategy_class = strategy_map.get(args.hpo_strategy)
        if strategy_class is None:
            raise ValueError(f"Unknown HPO strategy: {args.hpo_strategy}")

        hpo_strategy = strategy_class(
            solver_name=args.solver, cpm_model=cpm_model, max_tries=args.max_tries,
            round_type_strategy=round_time_strategy,
            global_time_splitting_strategy=global_time_strategy,
            timeout_evolution_strategy=timeout_evolution_strategy,
            all_configs=tunable_params, defaults=default_params,
            xpath=args.input if "xcsp" in args.solver else None
        )

        return PSA(cpm_model, hpo_strategy)
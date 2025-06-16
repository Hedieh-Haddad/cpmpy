# cpmpy/tools/probing_solving.py

# Import the abstract base class for strategies, which ensures a consistent interface.
from cpmpy.tools.psa.hpo_strategies import HPOStrategy
# Import the custom logger and enums for type clarity.
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.enum import RoundTimeType, TimeoutEvolution
from cpmpy import Model


class PSA:
    """
    Probe and Solve Algorithm (PSA) Orchestrator.

    This is the main engine of the tuning framework. It takes a CPMpy model and a
    pre-configured HPO strategy. Its primary role is to execute the tuning loop,
    manage the different phases (probing, solving), and provide a clean interface
    to access the final results.
    """

    def __init__(self, cpm_model: Model, hpo_strategy: HPOStrategy):
        # Store the model and the fully configured strategy object.
        self._cpm_model = cpm_model
        self._hpo_strategy = hpo_strategy
        # This will hold the final solver instance after all phases are complete,
        # allowing easy access to the final status, runtime, and objective.
        self.final_solver = None

    def tune(self, time_limit: int, max_tries: int = 100):
        """
        Executes the full tuning process.

        1. Initializes the HPO strategy (sets up timers, runs initial configs, etc.).
        2. Enters the main probing loop, which continues as long as the time strategy allows.
        3. Within the loop, it runs one probing trial and updates the round timeout.
        4. After the loop, it runs the final solving phase with the best-found parameters.
        5. It stores the final solver instance for property access.
        6. Finally, it returns the best found hyperparameter configuration.
        """
        self._hpo_strategy.initialize(time_limit=time_limit, max_tries=max_tries)

        while self._hpo_strategy.probing_should_continue():
            self._hpo_strategy.probing_phase()
            self._hpo_strategy.update_current_timeout()

        self._hpo_strategy.solving_phase()

        # Store the final solver instance so its properties can be accessed.
        self.final_solver = self._hpo_strategy._solver
        return self._hpo_strategy.finalize()

    # The following properties provide a clean public API to access the results
    # of the tuning process from the `psa` object in the main run script.

    @property
    def hpo_strategy_name(self):
        """Returns the name of the HPO strategy being used."""
        return self._hpo_strategy.name if self._hpo_strategy else "Unknown"

    @property
    def final_status(self):
        """Returns the exit status of the final solver run."""
        return self.final_solver.status().exitstatus if self.final_solver else "Not Run"

    @property
    def final_runtime(self):
        """Returns the runtime of the final solver run."""
        return self.final_solver.status().runtime if self.final_solver else 0.0

    @property
    def final_objective(self):
        """Returns the objective value of the final solver run."""
        return self.final_solver.objective_value() if self.final_solver else None

    @property
    def solving_time_budget(self):
        """Returns the time allocated for the final solving phase."""
        return self._hpo_strategy._global_time_splitting_strategy.solving_timeout


class PSABuilder:
    """
    A builder class for constructing a PSA object.
    This provides a fluent interface for setting up the PSA, though it's less used
    now that the more comprehensive PSAFactory exists.
    """

    def __init__(self, cpm_model: Model):
        self._cpm_model = cpm_model
        self._hpo_strategy: HPOStrategy | None = None

    def with_hpo_strategy(self, hpo_strategy: HPOStrategy):
        self._hpo_strategy = hpo_strategy
        return self

    def build(self):
        return PSA(self._cpm_model, self._hpo_strategy)


class PSAFactory:
    """
    A factory class responsible for creating and configuring the entire PSA pipeline
    based on command-line arguments. This decouples the main run script from the
    specific implementations of different strategies, making the system more modular.
    """

    @staticmethod
    def create_psa_from_cli(args, cpm_model):
        # These imports are done inside the method to avoid circular dependencies.
        from cpmpy.tools.psa.hpo_strategies import BayesianOptimizationStrategy, HammingDistanceNoPSAStrategy
        from cpmpy.tools.psa.time_strategies import (
            PercentageTuningGlobalTimeoutStrategy,
            StaticRoundTimeStrategy, FirstRuntimeRoundTimeStrategy,
            StaticTimeoutEvolutionStrategy, DynamicGeometricTimeoutEvolutionStrategy,
            DynamicLubyTimeoutEvolutionStrategy
        )
        import json

        # Load hyperparameter definitions from the specified JSON file.
        if args.tuning_file:
            with open(args.tuning_file, 'r') as f:
                tuning_config = json.load(f)
            tunable_params = tuning_config.get("tunable_params", {})
            default_params = tuning_config.get("default_params", {})
        else:
            log("Warning: No tuning file provided. Using empty hyperparameter space.", "warning")
            tunable_params, default_params = {}, {}

        # Configure time splitting strategy based on the chosen HPO method.
        # The 'hamming' (NoPSA) strategy uses the full time limit for its single tuning loop.
        if args.hpo_strategy == "hamming":
            log("Using 'hamming' (No PSA) strategy, forcing global time to 100% for tuning.", "info")
            time_split_percent = 1.0
        else:  # 'bayesian' uses the probe/solve split percentage from the command line.
            time_split_percent = args.percent

        global_time_strategy = PercentageTuningGlobalTimeoutStrategy(
            global_timeout=args.global_time_limit,
            percent=time_split_percent
        )

        # Configure round time and evolution strategies based on CLI arguments.
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

        # Map the CLI strategy name (e.g., "bayesian") to the correct class.
        strategy_map = {
            "bayesian": BayesianOptimizationStrategy,
            "hamming": HammingDistanceNoPSAStrategy,
        }

        strategy_class = strategy_map.get(args.hpo_strategy)
        if strategy_class is None:
            raise ValueError(f"Unknown HPO strategy: {args.hpo_strategy}")

        # Instantiate the chosen HPO strategy with all its configured components.
        hpo_strategy = strategy_class(
            solver_name=args.solver, cpm_model=cpm_model, max_tries=args.max_tries,
            round_type_strategy=round_time_strategy,
            global_time_splitting_strategy=global_time_strategy,
            timeout_evolution_strategy=timeout_evolution_strategy,
            all_configs=tunable_params, defaults=default_params,
            xpath=args.input if "xcsp" in args.solver else None
        )

        # Finally, create and return the main PSA object with the configured strategy.
        return PSA(cpm_model, hpo_strategy)
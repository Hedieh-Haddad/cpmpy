"""
Timeout strategy module for probing and solving in parameter tuning.

This module provides a set of strategies to control how time is allocated during the tuning process
of solver configurations. It implements strategies to:

- Decide how to divide the global timeout between probing and solving.
- Set how long each round of configuration evaluation should take.
- Evolve the round timeout dynamically using sequences (e.g., geometric or Luby).

The overall idea is to:
1. Split the total time budget (`global_timeout`) into `probe_timeout` and `solving_timeout`.
2. Test various solver configurations during `probe_timeout`, each for `round_timeout` seconds.
3. Select the best configuration found and use it to solve the full instance in `solving_timeout`.

================
List of classes
================

.. autosummary::
    :nosignatures:

    RoundTimeStrategy
    StaticRoundTimeStrategy
    DynamicRoundTimeStrategy
    NoneRoundTimeStrategy
    TimeoutEvolutionStrategy
    StaticTimeoutEvolutionStrategy
    DynamicGeometricTimeoutEvolutionStrategy
    DynamicLubyTimeoutEvolutionStrategy
    TuningGlobalTimeoutStrategy
    PercentageTuningGlobalTimeoutStrategy
    TimeStrategyFactory
"""

from abc import ABC, abstractmethod

from cpmpy.tools.psa.enum import TimeType, TimeoutEvolution, RoundTimeType
from cpmpy.tools.psa.log import log
from cpmpy.tools.psa.utils import geometric_sequence, luby_sequence
from timeit import default_timer as timer


class RoundTimeStrategy(ABC):
    """
    Abstract base class for defining how to compute the timeout for each round during probing.

    A round timeout is the amount of time given to each configuration tested during the probing phase.
    """

    def __init__(self, solver, default_config):
        self.solver = solver
        self.default_config = default_config
        self._round_timeout = None
        self._runtime = 0

    @abstractmethod
    def init(self, time_limit=None):
        """
        Initialize the strategy, possibly using the probe timeout or the model itself.

        Args:
            time_limit (float, optional): Global time limit (used for 'None' strategy).
        """
        pass

    @property
    def runtime(self):
        """Return the current probe timeout (must be initialized first)."""
        assert self._runtime is not None
        return self._runtime

    @property
    def round_timeout(self):
        """Return the current round timeout (must be initialized first)."""
        assert self._round_timeout is not None
        return self._round_timeout


class StaticRoundTimeStrategy(RoundTimeStrategy):
    """
    Strategy where each configuration round gets a fixed time budget (e.g., 5 seconds).
    """

    def init(self, time_limit=None):
        self._round_timeout = 5


class FirstRuntimeRoundTimeStrategy(RoundTimeStrategy):
    """
    Strategy that estimates round timeout based on runtime of default config under probe timeout.

    A single run with the default configuration is executed using the entire probe timeout.
    The observed runtime is used as the round timeout for future configurations.
    """

    def init(self, time_limit=None):
        self.solver.solve(time_limit=time_limit, **self.default_config)
        runtime = self.solver.status().runtime
        self._round_timeout = runtime
        self._runtime = runtime


# class NoneRoundTimeStrategy(RoundTimeStrategy):
#     """
#     Strategy where the round timeout is set explicitly to the time limit.
#     """
#
#     def init(self, time_limit=None):
#         self._round_timeout = time_limit
#         self._probe_timeout = time_limit


class TimeoutEvolutionStrategy(ABC):
    """
    Abstract base class for evolving the round timeout across iterations (probing rounds).
    """

    @abstractmethod
    def evolve(self, current_time_limit, timeout_list=None):
        """
        Compute the next round timeout based on current state.

        Args:
            current_time_limit (float): The timeout used in the last round.
            timeout_list (list, optional): List of all previous timeouts.

        Returns:
            float: The new round timeout to use.
        """
        pass


class StaticTimeoutEvolutionStrategy(TimeoutEvolutionStrategy):
    """Never changes the timeout: each round uses the same duration."""

    def evolve(self, current_time_limit, timeout_list=None):
        return current_time_limit


class DynamicGeometricTimeoutEvolutionStrategy(TimeoutEvolutionStrategy):
    """
    Increases timeout geometrically: timeout(i+1) = geometric_sequence(timeout(i)).
    """

    def evolve(self, current_time_limit, timeout_list=None):
        return geometric_sequence(current_time_limit)


class DynamicLubyTimeoutEvolutionStrategy(TimeoutEvolutionStrategy):
    """
    Uses the Luby sequence to determine the next timeout.
    Requires a list to track timeout history.
    """

    def evolve(self, current_time_limit, timeout_list=None):
        index = len(timeout_list)
        return luby_sequence(index, current_time_limit, timeout_list)


class TuningGlobalTimeoutStrategy(ABC):
    """
    Abstract base class for splitting the total time budget between probing and solving.

    For example, if the total timeout is 1200 seconds:
        - 240s (20%) could be allocated to probing different configurations
        - 960s (80%) left for solving using the best found config
    """

    def __init__(self, global_timeout):
        self._global_timeout = global_timeout
        self._probe_timeout = None
        self._solving_timeout = None
        self._start_time = None
        self._round_counter = 0

    @property
    def global_timeout(self):
        return self._global_timeout

    @property
    def probe_timeout(self):
        return self._probe_timeout

    @property
    def solving_timeout(self):
        return self._solving_timeout

    @abstractmethod
    def init(self):
        """
        Initialize timeout splitting logic (must set self._probe_timeout and _solving_timeout).
        """
        pass

    def update_probe_timeout(self, probe_timeout):
        """
        Update the probe timeout if needed.
        """
        self._probe_timeout = probe_timeout
        self._solving_timeout = self.global_timeout - self.probe_timeout

    def update_global_timeout(self, global_timeout):
        """
        Update the global timeout if needed.
        """
        if global_timeout is not None:
            self._global_timeout = global_timeout

    def update_max_tries(self, max_tries: int):
        pass

    def start_probe_phase(self):
        self._start_time = timer()
        self._round_counter = 0

    def update_probe_phase(self):
        self._round_counter += 1

    def probe_phase_must_finish(self, current_timeout: int):
        result = self.elapsed_time + current_timeout >= self.probe_timeout
        log(str(result),"debug")
        return self.elapsed_time + current_timeout >= self.probe_timeout

    @property
    def elapsed_time(self):
        return timer() - self._start_time
    @property
    def round_counter(self):
        return self._round_counter

class PercentageTuningGlobalTimeoutStrategy(TuningGlobalTimeoutStrategy):
    """
    Splits the global timeout into fixed percentage for probing.

    For example, 20% for probing and 80% for solving.
    """

    def __init__(self, global_timeout, percent=0.2):
        super().__init__(global_timeout)
        self._percent = percent

    def init(self):
        self._probe_timeout = self._global_timeout * self._percent
        self._solving_timeout = self._global_timeout - self._probe_timeout


class NoLimitTuningGlobalTimeoutStrategy(TuningGlobalTimeoutStrategy):

    def __init__(self, global_timeout):
        super().__init__(global_timeout)
        self._max_tries = 0


    def init(self):
        self._probe_timeout = self.global_timeout
        self._solving_timeout = self.global_timeout

    def update_max_tries(self, max_tries: int):
        self._max_tries = max_tries

    def probe_phase_must_finish(self, current_timeout: int):
        return super().probe_phase_must_finish(current_timeout) or self._round_counter >= self._max_tries



class TimeStrategyFactory:
    """
    Factory for instantiating timeout and round time strategies based on enum values.
    """

    @staticmethod
    def create_round_time_splitting_strategy(strategy_type: RoundTimeType, solver, default_config):
        """
        Create round timeout strategy depending on the strategy type.

        Args:
            strategy_type (TimeType): Type of strategy (STATIC, DYNAMIC, NONE).
            solver: Solver instance.
            default_config (dict): Default parameters.

        Returns:
            RoundTimeStrategy: Strategy instance.
        """
        if strategy_type == RoundTimeType.STATIC:
            return StaticRoundTimeStrategy(solver, default_config)
        elif strategy_type == RoundTimeType.FIRST_RUNTIME:
            return FirstRuntimeRoundTimeStrategy(solver, default_config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    @staticmethod
    def create_global_time_splitting_strategy(strategy_type: TimeType, global_timeout: int, percent: float = 0.2):
        """
        Create global timeout splitting strategy depending on the strategy type.

        Args:
            strategy_type (TimeType): Type of strategy (STATIC, DYNAMIC).
            global_timeout (int): Global timeout value.
            percent (float): Percentage for probing.

        Returns:
            TuningGlobalTimeoutStrategy: Strategy instance.
        """
        if strategy_type == TimeType.PERCENT:
            return PercentageTuningGlobalTimeoutStrategy(global_timeout, percent)
        elif strategy_type == TimeType.DYNAMIC:
            return NoLimitTuningGlobalTimeoutStrategy(global_timeout)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    @staticmethod
    def create_timeout_evolution_strategy(strategy_type: TimeoutEvolution):
        """
        Create timeout evolution strategy depending on the strategy type.

        Args:
            strategy_type (TimeoutEvolution): Type of evolution (STATIC, GEOMETRIC, LUBY).

        Returns:
            TimeoutEvolutionStrategy: Strategy instance.
        """
        if strategy_type == TimeoutEvolution.STATIC:
            return StaticTimeoutEvolutionStrategy()
        elif strategy_type == TimeoutEvolution.DYNAMIC_GEOMETRIC:
            return DynamicGeometricTimeoutEvolutionStrategy()
        elif strategy_type == TimeoutEvolution.DYNAMIC_LUBY:
            return DynamicLubyTimeoutEvolutionStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

import enum


class HPOType(enum.Enum):
    """Hyperparameter optimization types."""
    GRID_SEARCH = "Grid"
    BAYESIAN_SEARCH = "Bayesian"
    FREE_SEARCH = "Free"
    HAMMING_SEARCH = "Hamming"


class StopCondition(enum.Enum):
    """Stop conditions for hyperparameter optimization."""
    TIMEOUT = "Timeout"
    FIRST_SOLUTION = "FirstSolution"

class TimeoutEvolution(enum.Enum):
    """Timeout evolution strategies."""
    STATIC = "Static"
    DYNAMIC_GEOMETRIC = "DynamicGeometric"
    DYNAMIC_LUBY = "DynamicLuby"


class TimeType(enum.Enum):
    """Round timeout types."""
    PERCENT = "Percent"
    DYNAMIC = "Dynamic"


class RoundTimeType(enum.Enum):
    """Round timeout types."""
    STATIC = "Static"
    FIRST_RUNTIME = "FirstRuntime"
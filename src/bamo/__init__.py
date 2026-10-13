"""Budget-adaptive metaheuristic optimization (BAMO) for ECG classification.

Core search code (``bamo.search``, ``bamo.metrics``, ``bamo.analysis``) depends only on
NumPy so it can be verified on cheap synthetic objectives. Data loading, training and
library comparators import their heavy dependencies lazily.
"""

__version__ = "0.1.0"

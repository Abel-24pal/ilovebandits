"""Centralized test constants for the ilovebandits test suite.

This module contains essential constant values used across the test suite.
"""

# ============================================================================
# Random State & Reproducibility
# ============================================================================

# Primary random seed for reproducible tests
RANDOM_SEED = 42

# Random state for data generation (sklearn, pandas)
RANDOM_STATE = 42


# ============================================================================
# Simulation Parameters
# ============================================================================

# Default number of iterations for simulations
TEST_ITERATIONS_DEFAULT = 1000

# Minimum iterations before starting agent training in simulations
TEST_MIN_ITES_TO_TRAIN = 30

# Update frequency for agent training (1 = every iteration, 28 = every 28th iteration)
TEST_UPDATE_FACTOR = 28

# Delay values for testing reward delays
TEST_REWARD_DELAY_NONE = 0
TEST_REWARD_DELAY_SMALL = 10

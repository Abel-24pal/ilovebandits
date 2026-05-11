"""Global constants and default configuration values for ilovebandits."""

# ============================================================================
# Agent Behavior Constants
# ============================================================================

# Minimum number of unique rewards per arm to avoid NotEnoughRewardsPerArmError
MIN_REWARDS_PER_ARM = 2

# Minimum number of samples per arm before ignoring the MIN_REWARDS_PER_ARM check
# Once this threshold is reached, the NotEnoughRewardsPerArmError will not be raised
MIN_SAMPLES_TO_IGNORE_ARM = 100

# Default number of initial rounds where the agent takes random actions
# This allows for exploration before exploitation begins
DEFAULT_N_ROUNDS_RANDOM = 200


# ============================================================================
# Exploration/Exploitation Parameters
# ============================================================================

# Default epsilon value for epsilon-greedy strategies
# Represents the probability of taking a random action (exploration)
DEFAULT_EPSILON = 0.1

# Default variance parameter for Thompson Sampling and UCB agents
# Controls the amount of exploration vs exploitation
DEFAULT_VPAR = 1.0

# Default number of samples for frequency estimation in Thompson Sampling
DEFAULT_SAMPLES_FOR_FREQ_EST = 100

# Default number of samples for frequency estimation in MAB Thompson Sampling
# Higher value for MAB as it typically needs more precision
DEFAULT_MAB_SAMPLES_FOR_FREQ_EST = 100000


# ============================================================================
# Random Forest Default Parameters
# ============================================================================

# Default criterion for RandomForest classifier
# "log_loss" provides accurate probability estimates
DEFAULT_RF_CRITERION_CLASSIFIER = "log_loss"

# Default minimum samples per leaf for tree-based agents
DEFAULT_RF_MIN_SAMPLES_LEAF = 20

# Default maximum depth for tree-based agents
DEFAULT_RF_MAX_DEPTH = 3

# Default random state for reproducibility
DEFAULT_RF_RANDOM_STATE = 42


# ============================================================================
# Simulation Defaults
# ============================================================================

# Default number of iterations for simulations
DEFAULT_SIMULATION_ITERATIONS = 1000

# Default reward delay (0 = no delay)
DEFAULT_REWARD_DELAY = 0


# ============================================================================
# Model Configuration Defaults
# ============================================================================

# Default flag for using one model per arm vs single shared model
DEFAULT_ONE_MODEL_PER_ARM = True

# Default divisor for bootstrap frequency in BootStrapConAgent
# If 1: always bootstrap when take_agent_action() is called
# If 2: bootstrap half the times
# If 3: bootstrap one third of the times, etc.
DEFAULT_DIVISOR_BOOTSTRAP = 1

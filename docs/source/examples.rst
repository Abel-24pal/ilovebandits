Examples
=============

Installation/Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

How to use an agent
**************************************************
.. code-block:: python

    """This example demonstrates how to initialize and use a bandit agent with the ilovebandits package.
    """
    from sklearn.linear_model import LinearRegression
    RANDOM_SEED = 42
    
    arms = 4
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=LinearRegression(),
        n_rounds_random=50,
        epsilon=0.1,
        one_model_per_arm=True,
        rng_seed=RANDOM_SEED,
    )
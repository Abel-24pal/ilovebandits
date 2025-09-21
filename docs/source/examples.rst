Examples
=============

Installation/Usage:
*******************
As the package is published in PyPI, you can install it using pip. In the environment you want to install it, just run:

.. code-block:: bash

   $ (sudo) pip install ilovebandits


You can use other package managers like conda or poetry, but you may need to specify that the package should be installed from PyPI if it is not the default channel.

How to use an agent
**************************************************
.. code-block:: python

    """This example demonstrates how to initialize and use a bandit agent with the ilovebandits package.
    """
    from sklearn.ensemble import RandomForestRegressor
    RANDOM_SEED = 42

    arms = 4
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=RandomForestRegressor(random_state=RANDOM_SEED),
        n_rounds_random=50,
        epsilon=0.1,
        one_model_per_arm=True,
        rng_seed=RANDOM_SEED,
    )

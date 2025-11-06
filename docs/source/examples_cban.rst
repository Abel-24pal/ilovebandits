Examples Contextual Bandits
===============================

How to initialize a basic contextual agent
**************************************************
In the example below, you can see an example of how to initialize and use an agent from the package.

.. code-block:: python

    """This example demonstrates how to initialize and use a bandit agent with the ilovebandits package."""
    from ilovebandits.agents import EpsGreedyConAgent
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

For the ``base_estimator``, you can use any regressor or classifier from scikit-learn or any other library that follows the scikit-learn interface.
In the example below, we use a `RandomForestRegressor`_ from scikit-learn.
However, you can use any other state-of-the-art ML models such as `XGBoost`_ or `LightGBM`_ with the corresponding scikit-learn wrapper.

.. _XGBoost: https://xgboost.readthedocs.io/en/latest/
.. _LightGBM: https://lightgbm.readthedocs.io/en/latest/

The ``epsilon`` parameter controls the exploration-exploitation trade-off for the :class:`ilovebandits.mab.agents.EpsGreedyConAgent`. A value of 0.1 means that the agent will explore 10% of the time and exploit 90% of the time.
The ``one_model_per_arm`` parameter indicates whether to use a disjoint model for each arm or a hybrid model for all arms.
Usually, disjoint models should be used, but as stated in research references such as `Tree Ensembles for Contextual Bandits`_, hybrid models make sense and can be useful when the base model is based on decision trees.
For instance, `RandomForestRegressor`_.

Once the agent is initialized, you can use it to select an action and update it with the observed reward.

How to initialize an advanced contextual agent
**************************************************
ilovebandits library provides advanced contextual agents such as :class:`ilovebandits.agents.RandomForestTsAgent` and :class:`ilovebandits.agents.RandomForestUcbAgent`
where UCB and Thompson Sampling are generalized for tree ensembles. They are based on the most recent state-ofthe-art algorithms (See `Tree Ensembles for Contextual Bandits`_ for additional details).

In the example below, you can see how to initialize a :class:`ilovebandits.agents.RandomForestTsAgent` (class that implements Thompson Sampling combined with Random Forest):

.. code-block:: python

    """This example demonstrates how to initialize and use a RandomForestTsAgent with the ilovebandits package."""
    from ilovebandits.agents import RandomForestTsAgent
    from sklearn.ensemble import RandomForestRegressor
    RANDOM_SEED = 42

    arms = 4
    rf_model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=3,
        max_depth=3,
        criterion="squared_error",
        max_samples=None, # We recommend setting this to None for better uncertainty estimates with the assumptions of RandomForestTsAgent
        random_state=RANDOM_SEED,
    )
    rfts_agent = RandomForestTsAgent(
        arms=arms,
        vpar=1, # parameter that controls Thompson Sampling exploration.
        base_model=rf_model, # Use here a RandomForestRegressor or RandomForestClassifier instance
        rng_seed=RANDOM_SEED,
        one_model_per_arm=False,
        samples_for_freq_est=100,
        n_rounds_random=0,
        min_rewards_per_arm=2,
    )

For the ``base_model``, you should use a `RandomForestRegressor`_ or `RandomForestClassifier`_ from scikit-learn.
Use only `RandomForestClassifier`_ if the expected rewards will be in :math:`\{0, 1\}`. Otherwise use `RandomForestRegressor`_.

The ``vpar`` parameter controls Thompson Sampling exploration. Higher values lead to more exploration as the posterior distributions will have higher variances.

The ``samples_for_freq_est`` parameter controls the number of samples used for probability estimation in the action chosen for the agent.
Use ``samples_for_freq_est=1`` if you are not interested in estimating the probability of the chosen action.

The min_rewards_per_arm parameter controls the minimum number of unique rewards per arm you should have in order to do the first update of the agent.
If the condition is not satisfied, the :class:`ilovebandits.exceptions.NotEnoughRewardsPerArmError` will be raised.
This is a guardrail to avoid updating the agent with no rewards for some arms in the first iteration which could lead to problematic behaviors for future selection of those arms.
Following the same philosofy, we have the min_samples_to_ignore_arm parameter which fixs a minimum number of samples per arm to allow the first update of the agent. If not,
:class:`ilovebandits.exceptions.NotEnoughRewardsPerArmError` will be raised.
The parameters ``min_rewards_per_arm`` and ``min_samples_to_ignore_arm`` are available for all contextual agents such as :class:`ilovebandits.mab.agents.EpsGreedyConAgent`
as they are useful to avoid problems when the initial data used to update the agent is very small.

Other parameters like ``one_model_per_arm`` and ``n_rounds_random`` have the same meaning as in the previous example.

In the example below, you can see how to initialize a :class:`ilovebandits.agents.RandomForestUcbAgent` (class that implements UCB combined with Random Forest):

.. code-block:: python

    """This example demonstrates how to initialize and use a RandomForestTsAgent with the ilovebandits package."""
    from ilovebandits.agents import RandomForestUcbAgent
    from sklearn.ensemble import RandomForestRegressor
    RANDOM_SEED = 42

    arms = 4
    rf_model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=3,
        max_depth=3,
        criterion="squared_error",
        max_samples=None, # We recommend setting this to None for better uncertainty estimates with the assumptions of RandomForestTsAgent
        random_state=RANDOM_SEED,
    )
    rfts_agent = RandomForestUcbAgent(
        arms=arms,
        vpar=1, # parameter that controls UCB exploration increasing uncertainty estimation.
        base_model=rf_model, # Use here a RandomForestRegressor or RandomForestClassifier instance
        rng_seed=RANDOM_SEED,
        one_model_per_arm=False,
        n_rounds_random=10,
    )

The ``vpar`` controls UCB exploration increasing uncertainty estimation. Higher values lead to more exploration.
Other parameters have the same meaning as previously discussed.

.. _RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
.. _RandomForestRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
.. _Tree Ensembles for Contextual Bandits: https://research.chalmers.se/publication/545472/file/545472_Fulltext.pdf

Arm selection and updates in contextual agents
**************************************************
The following example shows how to update a contextual agent with new data using :meth:`ilovebandits.agents.EpsGreedyConAgent.update_agent` and take an action based on the current model using :meth:`ilovebandits.agents.EpsGreedyConAgent.take_action`.

.. code-block:: python

    """This example demonstrates how to update an agent and take an action with the ilovebandits package."""
    import numpy as np
    from ilovebandits.agents import EpsGreedyConAgent
    from sklearn.ensemble import RandomForestRegressor
    RANDOM_SEED = 42

    # We update the agent with a new batch of samples. Imagine the following training data:

    # Array with  arms selected for each sample
    a_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1,
        1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3])

    # Array with  rewards obtained for each sample
    r_train = np.array([ -3.,   4., -11.,  10.,  13.,  11.,  47.,  24.,  20.,  35.,  76.,
            84.,   4.,   3.,  13.,   5.,  -6.,   8., -22.,  20.,  26.,  22.,
            94.,  48.,  40.,  70., 152., 168.,   8.,   6.,  26.,  10.,  -9.,
            12., -33.,  30.,  39.,  33., 141.,  72.,  60., 105., 228., 252.,
            12.,   9.,  39.,  15.])

    # Array with feature values (three feature columns) obtained for each sample
    c_train = np.array([[ 1, -1,  2],
        [ 2,  3,  4],
        [ 3, -3,  8],
        [ 4,  8, 10],
        [ 1, -1,  2],
        [ 2,  3,  4],
        [ 3, -3,  8],
        [ 4,  8, 10],
        [ 1, -1,  2],
        [ 2,  3,  4],
        [ 3, -3,  8],
        [ 4,  8, 10],
        [ 1, -1,  2],
        [ 2,  3,  4],
        [ 3, -3,  8],
        [ 4,  8, 10],
        [ 2, -2,  4],
        [ 4,  6,  8],
        [ 6, -6, 16],
        [ 8, 16, 20],
        [ 2, -2,  4],
        [ 4,  6,  8],
        [ 6, -6, 16],
        [ 8, 16, 20],
        [ 2, -2,  4],
        [ 4,  6,  8],
        [ 6, -6, 16],
        [ 8, 16, 20],
        [ 2, -2,  4],
        [ 4,  6,  8],
        [ 6, -6, 16],
        [ 8, 16, 20],
        [ 3, -3,  6],
        [ 6,  9, 12],
        [ 9, -9, 24],
        [12, 24, 30],
        [ 3, -3,  6],
        [ 6,  9, 12],
        [ 9, -9, 24],
        [12, 24, 30],
        [ 3, -3,  6],
        [ 6,  9, 12],
        [ 9, -9, 24],
        [12, 24, 30],
        [ 3, -3,  6],
        [ 6,  9, 12],
        [ 9, -9, 24],
        [12, 24, 30]])

.. code-block:: python

    """This example demonstrates how to update an agent and take an action with the ilovebandits package."""
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=RandomForestRegressor(random_state=RANDOM_SEED),
        n_rounds_random=5,
        epsilon=0.1,
        one_model_per_arm=True,
        rng_seed=RANDOM_SEED,
    )

    #### UPDATE AGENT ######
    eps_agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)

    # check number of updates of the agent
    print(eps_agent.update_agent_counts)
    # Check agent hybrid model if option selected:
    print(eps_agent.model)
    # Check agent disjoint models if option selected:
    print(eps_agent.models)
    # Check number of features used by the agent
    print(eps_agent.nfeats)


    ########### PREDICT AGENT ##########
    dummy_context = np.ones((1, eps_agent.nfeats))  # Create a dummy context with the appropriate number of features
    (sel_arm, prob_sel_arm) = eps_agent.take_action(context=dummy_context)
    print(f"Selected arm: {sel_arm}, Probability of selected arm to be chosen: {prob_sel_arm}")

    # Do additional 10 arm selections to finish n_rounds_random and start epsilon-greedy selections (we just imagine the same dummy_context for simplicity)
    for i in range(10):
        (sel_arm, prob_sel_arm) = eps_agent.take_action(context=dummy_context)
        print(f"Selected arm: {sel_arm}, Probability of selected arm to be chosen: {prob_sel_arm}")

Fast agent updates for RandomForestTsAgent and RandomForestUcbAgent agents
*******************************************************************************
The classes :class:`ilovebandits.agents.RandomForestTsAgent` and :class:`ilovebandits.agents.RandomForestUcbAgent`
implement fast updates once the tree ensembles are created. This fast updates do not change the tree structures,
but only update the leaf statistics. This is very useful when the number of samples is very large and we want to avoid
the computational cost of retraining the tree ensembles from scratch at each update.
Here an example of how to use this fast update feature and the time savings it provides:

.. code-block:: python

    """This example demonstrates how to use fast updates with RandomForestTsAgent with the ilovebandits package."""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from ilovebandits.agents import RandomForestTsAgent
    from sklearn.ensemble import RandomForestRegressor
    RANDOM_SEED = 42

    random_state, n_features, n_samples = 42, 20, 75_000

    # Generate a regression dataset.
    x, y = make_regression(
        n_samples=n_samples, n_features=n_features, noise=1, random_state=random_state
    )

    # Use last column as arm column. Convert values to integers in the range [0, n_features-1].
    q1 = pd.DataFrame(x).describe().loc["25%", n_features - 1]
    q2 = pd.DataFrame(x).describe().loc["50%", n_features - 1]
    q3 = pd.DataFrame(x).describe().loc["75%", n_features - 1]

    x[:, -1] = np.where(
        x[:, -1] < q1, 0, np.where(x[:, -1] < q2, 1, np.where(x[:, -1] < q3, 2, 3))
    )

    x_train, x_train2, y_train, y_train2 = train_test_split(
        x, y, test_size=0.8, random_state=42
    )

    c_train = x_train[:, :-1]  # Contextual features
    a_train = x_train[:, -1].astype(int)  # Arm selected

    c_train2 = x_train2[:, :-1]  # Contextual features
    a_train2 = x_train2[:, -1].astype(int)  # Arm selected

    arms = 4
    rf_model = RandomForestRegressor(
        n_estimators=100,
        criterion="squared_error",
        max_samples=None, # We recommend setting this to None for better uncertainty estimates with the assumptions of RandomForestTsAgent
        random_state=RANDOM_SEED,
    )
    rfts_agent = RandomForestTsAgent(
        arms=arms,
        vpar=1, # parameter that controls Thompson Sampling exploration.
        base_model=rf_model, # Use here a RandomForestRegressor or RandomForestClassifier instance
        rng_seed=RANDOM_SEED,
        one_model_per_arm=False,
        samples_for_freq_est=100,
        n_rounds_random=0,
    )

    rfts_agent.update_agent(c_train=c_train, a_train=a_train, r_train=y_train)

.. code-block:: python

    %%time
    # Perform a partial fast update with new data.
    # Note: this method automatically takes into consideration the previous training data.
    rfts_agent.partial_fast_update(c_train=c_train2, a_train=a_train2, r_train=y_train2)

.. code-block:: python

    %%time
    # Note: this method does not take into consideration the previous training data,
    # so we need to concatenate previous data.
    rfts_agent.update_agent(
        c_train=np.concatenate([c_train2, c_train], axis=0),
        a_train=np.concatenate([a_train2, a_train], axis=0),
        r_train=np.concatenate([y_train2, y_train], axis=0)
    )

If you compare the time taken by both methods, you will see that the ``partial_fast_update`` method is significantly faster than the ``update_agent`` method.
For this toy example, in the machine used for testing, the ``partial_fast_update`` method took around 2 seconds, while the ``update_agent`` method took around 3 minutes.
(We used a standard PC for personal usage)


Perform a simulation in a given environment
**************************************************
It is important to test the performance of the agent in a given environment. For that purpose,
the :class:`ilovebandits.sim.SimContBandit` is created.
It accepts 4 parameters:

- ``agent``: an instance of a contextual agent from ilovebandits.agents
- ``model_env``: an instance of a contextual bandit environment from ilovebandits.data_bandits
- ``min_ites_to_train``: minimum number of iterations to start training the agent
- ``update_factor``: if 1, it updates the model every iteration, if 2, it updates every two iterations, etc.

For the ``model_env``, we can use the :class:`ilovebandits.data_bandits.base.DataBasedBanditFromPandas` class,
which allows us to create a contextual bandit environment from a pandas DataFrame.
This class transforms a classification dataset into a contextual bandit environment.
This is the same technique employed in research papers such as `Neural Thompson Sampling`_.
It assumes the last column of the DataFrame is the target variable.
Each class is considered an arm. If the action taken matches the target label, it gives a reward of 1, 0 otherwise.

.. _Neural Thompson Sampling: https://scispace.com/pdf/neural-thompson-sampling-4yusqy8jg2.pdf

In addition, :class:`ilovebandits.data_bandits.base.DataBasedBanditFromPandas` allows us to simulate delayed rewards
if the reward_delay parameter is set to a value greater than 0.

The :class:`ilovebandits.data_bandits.utils.GenrlBanditDataLoader` allows us to easily load a commonly employed dataset
for benchmarking contextual bandit algorithms.

.. code-block:: python

    """This example demonstrates how to use SimContBandit() and DataBasedBanditFromPandas() classes."""
    from sklearn.ensemble import RandomForestClassifier
    from ilovebandits.agents import EpsGreedyConAgent
    from ilovebandits.data_bandits.base import DataBasedBanditFromPandas
    from ilovebandits.data_bandits.utils import GenrlBanditDataLoader
    from ilovebandits.sim import SimContBandit

    import pandas as pd

    RANDOM_SEED = 42
    RANDOM_STATE = 42

    reward_delay = 10
    iterations = 1000
    min_ites_to_train = 30  # minimum number of iterations to start training the agent
    update_factor = 28  # if 1, it updates the model every iteration, if 2, it updates every two iterations, etc.

    dataset_for_sims = GenrlBanditDataLoader().get_statlog_shuttle_data()

    model_env = DataBasedBanditFromPandas(
        df=dataset_for_sims,
        reward_delay=reward_delay,
        random_state=RANDOM_STATE,
    )
    narms = model_env.arms
    agent = EpsGreedyConAgent(
        arms=narms,
        base_estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        n_rounds_random=50,
        epsilon=0.1,
        one_model_per_arm=False,
        rng_seed=RANDOM_SEED,
        min_samples_to_ignore_arm=10,
    )

    simulator = SimContBandit(
        agent=agent,
        model_env=model_env,
        min_ites_to_train=min_ites_to_train,
        update_factor=update_factor,
    )

    res = simulator.simulate(iterations=iterations)

    #### You can obtain the rewards obtained by the agent at each iteration as a pandas DataFrame with thw followin coide line:
    # It contains 4 columns:
    #   -'ite': iteration the reward was received,
    #   -'arm': arm was selected,
    #   -'context': context features used,
    #   -'reward': reward received at 'ite'
    rew_agent = pd.DataFrame(res['rew_agent'])
    print(rew_agent)

    # You can also obtain a list of the actions selected by the agent at each iteration:
    print(res['actions'])

    # You can also obtain a list of the chosen action probabilities:
    print(res['prob_actions'])

    # You can also access the agent and model environment at the end of the simulation:
    print(res['agent'])
    print(res['model_env'])

    # For more information, please refer to the API docs.

Examples MAB
=============

How to initialize a multi armed bandit
**************************************************
In the example below, you can see an example of how to initialize and use an agent from the package.

.. code-block:: python

    """This example demonstrates how to initialize and use a bandit agent with the ilovebandits package.
    """
    from ilovebandits.mab.agents import EpsilonGreedyAgent, UCBAgent, TSAgent
    from ilovebandits.mab.q_estimators import QEstMean

    # Initialize a Q estimator. Here, we employ a sample average estimator with initial Q values of 0.
    q_estimator=QEstMean(arms=5, qvals_init=[0, 0, 0, 0, 0]) # Note: qvals_init indicates the initial reward estimation for each arm.

    # Initialize an epsilon-greedy agent, UCB agent, and Thompson Sampling agent.
    ep_greedy = EpsilonGreedyAgent(epsilon=0.1, q_estimator=q_estimator)
    ucb = UCBAgent(c=1, q_estimator=q_estimator)
    ts = TSAgent(arms=5)

As you can see, there are two main classes. The agent class, which defines the strategy to select arms, and the Q estimator class,
which defines how to estimate the rewards of each arm.

In the example above, we initialize an :class:`ilovebandits.mab.agents.EpsilonGreedyAgent` and :class:`ilovebandits.mab.agents.UCBAgent` agents
with a sample average estimator. See API documentation for other Q estimator and agent options.

In :class:`ilovebandits.mab.agents.TSAgent` (Thompson sampling agent), the reward estimator is built-in :class:`ilovebandits.mab.q_estimators.QThompSamp`.
By default, it assumes a :math:`Bernoulli(p)` reward distribution and a initial belief defined by a :math:`Beta(\alpha=1, \beta=1)` distribution.
This initial belief can be changed just instantiating :class:`ilovebandits.mab.agents.TSAgent` with different :math:`\alpha` and :math:`\beta` parameters using the arguments a_init and b_init.

In the :class:`ilovebandits.mab.agents.EpsilonGreedyAgent`, the ``epsilon`` parameter controls the exploration-exploitation trade-off.
A value of 0.1 means that the agent will explore 10% of the time and exploit 90% of the time.

In the :class:`ilovebandits.mab.agents.UCBAgent`, the ``c`` parameter controls the exploration-exploitation trade-off. The bigger the value of ``c``, the stronger the exploration the agent will do.

Once the agent is initialized, you can use it to select an action and update it with the observed reward.

Arm selection and updates MAB
**************************************************
The code below shows how to take an action and update the agent with the observed reward.
It introduces some important attributes and methods. For additional details, please refer to the API documentation.

.. code-block:: python

    """This example demonstrates how to take an action and update a bandit agent with the ilovebandits package.
    """
    from ilovebandits.mab.agents import EpsilonGreedyAgent
    from ilovebandits.mab.q_estimators import QEstMean

    # Initialize agent
    agent = EpsilonGreedyAgent(epsilon=0.1, q_estimator=q_estimator)

    # Select an action (arm) based on the agent's strategy and current estimates.
    sel_arm, count_sel, prob_sel_arm = agent.take_action()

    # Imagine that the selected arm sel_arm produced a reward equal to 1. To update the agent with this new information, you would do:
    reward = 1
    agent.q_estimator.estimate(reward=reward, action=sel_arm)  # Update the agent with the observed reward for the selected action.

    # Let's select another arm and update the agent again. Imagine, now we received again a reward equal to 1 for the selected arm.
    sel_arm, count_sel, prob_sel_arm = agent.take_action()
    reward = 1
    agent.q_estimator.estimate(reward=reward, action=sel_arm)

    # To access the current estimates of the rewards for each arm, you can use:
    print(agent.q_estimator.qvals)  # This will print the current estimated rewards for each arm.

    # To access the number of times each arm has been updated, you can use:
    print(agent.q_estimator.arm_count_updates)

    # To access the number of times each arm has been selected, you can use:
    print(agent.arm_count)

    # Sometime we want the agent to forget/reset everything it has learned. To do this, you can use:
    agent.reset_agent()  # This will reset the agent's internal state, including reward estimates and counts.

    print(agent.q_estimator.qvals)  # This will print the current estimated rewards for each arm.
    print(agent.q_estimator.arm_count_updates)
    print(agent.arm_count)

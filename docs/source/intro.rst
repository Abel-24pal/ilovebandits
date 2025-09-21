Introduction
============

``ilovebandits`` is a high-level Python package to work with contextual and multiarmed bandits with
some extended features. In essence, this package adds novel ensemble of trees techniques
for contextual bandits and the concept of composed rewards where you can feed the bandit with different type rewards in time.

The implementation has been developed in Python 3.
We are currently in a pre-alpha version focusing on adding core features and stabilizing architecture

Motivation
**********

There are a lot of potential business problems that can be solved with bandits, but current implementations lack some features that can be very useful to apply bandits in new scenarios. The main goal of this package is to provide a simple and easy-to-use interface to work with bandits, while also providing some advanced features that are not available in other packages.
Main industry limitations that this package address:

1. There are not many implementations of advanced contextual bandits that allow the use of ensemble of trees methods. This is a limitation because ensemble of trees methods are very powerful for tabular data. They need less data than Neural Networks and can capture richer pattern than the common linear methods employed in bandits.
2. The ability to define composed rewards. In many real-world scenarios, the reward is not a single value but a combination of different sub-rewards that can be observed at different times. For example, in a recommendation system, the final reward can be a combination of immediate clicks, time spent on the platform, and long-term user retention. Being able to define and use these sub-rewards can significantly improve the performance of the bandit algorithm and we can also update the bandit more often for each subpart as soon as it comes.

This packages concentrates on:

1. The use of ensemble of trees methods for contextual bandits.
2. The use of composed rewards where the bandit can be fed with different type of subrewards in time. Here, we define a long-term global reward, which we break down into smaller, shorter-term rewards that allow us to update the bandit much earlier than if we waited for the global reward.

These small updates also change the global reward estimate employed by the bandit to take the final decision”

This package is intended to provide a quick, as well as (hopefully) easy to undestand, way of getting a bandit simulations and core functions to create ready-to-use solutions for the industry.

Current version
***************

In latest version 0.1.0a1, we have:

- Multi-armed bandits (MAB):

  - Basic multi-armed bandit algorithms (e.g., Greedy, Epsilon-Greedy, Random, UCB, Thompson Sampling).
  - Basic reward estimators for MAB problem (e.g., Sample Average, Constant Step Size, ...)

- Contextual bandit algorithms:

  - Epsilon-Greedy with a big flexibility for the base model. For example, `XGBoost`_ or `LightGBM`_ can be used to learn the rewards from features.
  - BootStrapConAgent. It tries to simulate Thompson Sampling with the advantage of using any base model. For example, ensembles of trees can be used to learn the rewards from features (experimental)

- Environment simulator based on Statlog Shuttle dataset. (Dataset employed for many research papers on bandits. It can be obtained from `UCI Machine Learning Repository`_).

.. _XGBoost: https://xgboost.readthedocs.io/en/latest/
.. _LightGBM: https://lightgbm.readthedocs.io/en/latest/
.. _UCI Machine Learning Repository: https://archive.ics.uci.edu/

Core features planned to be added:

- Delayed rewards for MAB and contextual bandits.
- Composed rewards for MAB and contextual bandits.
- More contextual bandit algorithms with a special emphasis on ensemble of trees methods.
- Environment simulators

Limitations
***********

- Constant delays can be used for the simulators. Random non-constant rewards environments are only available for multi armed bandits.

- Although the interfacing operations of the :class:`ilovebandits.agents.BootStrapConAgent` is experimental and not yet fully tested. :meth:`ilovebandits.agents.BootStrapConAgent.update_agent` can be slow.

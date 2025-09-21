# ilovebandits

The implementation has been developed in Python 3. We are currently in a pre-alpha version focusing on adding core features and stabilizing architecture.

## Motivation

There are a lot of potential business problems that can be solved with bandits, but current implementations lack some features that can be very useful to apply bandits in new scenarios. The main goal of this package is to provide a simple and easy-to-use interface to work with bandits, while also providing some advanced features that are not available in other packages.

Main industry limitations that this package addresses:

1. **Advanced contextual bandits with ensembles of trees**
   There are not many implementations of advanced contextual bandits that allow the use of ensemble-of-trees methods. This is a limitation because ensemble methods are very powerful for tabular data. They need less data than neural networks and can capture richer patterns than the common linear methods employed in bandits.

2. **Ability to define composed rewards**
   In many real-world scenarios, the reward is not a single value but a combination of different sub-rewards that can be observed at different times.
   For example, in a recommendation system, the final reward can be a combination of immediate clicks, time spent on the platform, and long-term user retention.
   Being able to define and use these sub-rewards can significantly improve the performance of the bandit algorithm, and we can also update the bandit more often for each subpart as soon as it comes.

---

## This package concentrates on

1. The use of ensemble-of-trees methods for contextual bandits.
2. The use of composed rewards where the bandit can be fed with different types of sub-rewards in time. Here, we define a long-term global reward, which we break down into smaller, shorter-term rewards that allow us to update the bandit much earlier than if we waited for the global reward.

These small updates also change the global reward estimate employed by the bandit to take the final decision.

---

This package is intended to provide a quick, as well as (hopefully) easy-to-understand, way of running bandit simulations and core functions to create ready-to-use solutions for the industry.

# Installation

**Installation/Usage**: [Installation/Usage details in official docs](https://ilovebandits.readthedocs.io/en/latest/examples.html)

# Documentation

**Documentation**: [Official documentation of the project in readthedocs](https://ilovebandits.readthedocs.io)

# Current version content and limitations

[Current version content and limitations](https://ilovebandits.readthedocs.io/en/latest/intro.html)

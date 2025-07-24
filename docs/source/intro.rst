Introduction
============

``ilovebandits`` is a high-level Python package to work with contextual and multiarmed bandits with some extended features that are no available. In essence, this package is adds novel ensemble of trees techniques for contextual bandits and the concept of composed rewards where you can feed the bandit with different type rewards in time.

The current implementation has been developed in Python 3 ans scikit-learn >=1.5.1 and pandas >=2.2.2. It can work with older packages with maybe with minor modifications.

(see TEST LINK `here <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/build-the-docs.html>`_)

Motivation
**********

There are a lot of potential business problems that can be solved with bandits, but current implementations lack some features that can be very useful to apply bandits in new scenarios. The main goal of this package is to provide a simple and easy-to-use interface to work with bandits, while also providing some advanced features that are not available in other packages.

This package is intended to provide a quick, as well as (hopefully) easy to undestand, way of getting a bandit simulations and core functions to create ready-to-use solutions for the industry.

Limitations
***********

- Constant delays can be used for the simulators. Random non-constant rewards environments are only available for multi armed bandits.

- Although the interfacing operations of the :class:`ilovebandits.agents.BootStrapConAgent` is experimental and not yet tested. :meth:`ilovebandits.agents.BootStrapConAgent.update_agent` can be slow.
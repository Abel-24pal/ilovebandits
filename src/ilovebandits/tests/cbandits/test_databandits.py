"""Testing submodule data_bandits."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from src.ilovebandits.data_bandits.base import DataBasedBanditFromPandas
from src.ilovebandits.data_bandits.utils import GenrlBanditDataLoader

RANDOM_STATE = 42


@pytest.fixture
def shuttle_data():
    """Fixture to load the shuttle dataset. Output is a pandas DataFrame."""
    return GenrlBanditDataLoader().get_statlog_shuttle_data()


def test_get_shuttle_data(shuttle_data):
    """Test the data loader for the shuttle dataset."""
    assert isinstance(shuttle_data, pd.DataFrame)
    assert shuttle_data.shape == (43500, 10)


def test_databandit_from_pandas_with_shuttle_data(shuttle_data):
    """Test the DataBasedBanditFromPandas with the shuttle dataset.

    This test checks the initialization, context retrieval, and reward-taking functionality.
    """
    ds_ban = DataBasedBanditFromPandas(
        df=shuttle_data,
        reward_delay=0,
        random_state=RANDOM_STATE,
    )
    assert ds_ban.arms == 7
    assert ds_ban.nfeats == 9
    assert ds_ban.nobs == 43500
    assert ds_ban.idx_arms == list(range(7))
    assert ds_ban.idx == 0
    assert ds_ban.delay == 0

    # check the first context
    context = ds_ban.get_current_context()
    assert context.shape == (1, 9)
    npt.assert_array_equal(
        context,
        np.array([[53, 2, 88, 0, 52, -13, 35, 37, 2]]),
    )

    # check reward and delay according to random_state=42
    assert ds_ban.take_reward_of_arm(0) == (1, 0)
    assert ds_ban.take_reward_of_arm(1) == (0, 0)

    # check current value of the pointer
    assert ds_ban.idx == 2

    # check pointer's value after reset
    ds_ban.reset_env()
    assert ds_ban.idx == 0

    with pytest.raises(ValueError):
        ds_ban.take_reward_of_arm(7)  # Invalid arm

import pytest

from customer_sat.config import load_config
from customer_sat.synthetic import make_synthetic_data


@pytest.fixture(scope="session")
def df():
    return make_synthetic_data(1500, seed=1)


@pytest.fixture()
def cfg(tmp_path):
    c = load_config(None, {"output.dir": str(tmp_path / "out")})
    c.features.top_k_features = 15
    c.output.save_plots = False
    for name, m in c.train.models.items():          # keep tests fast
        m.params = {"max_depth": 3} if name == "dt" else {"n_estimators": 20}
    return c

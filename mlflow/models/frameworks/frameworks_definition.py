from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

FRAMEWORKS = {
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor,
        "default_params": {},
    },
    "LinearRegression": {
        "model": LinearRegression,
        "default_params": {},
    },
}

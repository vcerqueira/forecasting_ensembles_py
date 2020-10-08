from sklearn.ensemble \
    import (RandomForestRegressor,
            ExtraTreesRegressor,
            AdaBoostRegressor,
            GradientBoostingRegressor)
from sklearn.linear_model \
    import (Lasso,
            Ridge,
            ElasticNet)

METHODS = \
    dict(
        RandomForestRegressor=RandomForestRegressor,
        ExtraTreesRegressor=ExtraTreesRegressor,
        AdaBoostRegressor=AdaBoostRegressor,
        GradientBoostingRegressor=GradientBoostingRegressor,
        Lasso=Lasso,
        Ridge=Ridge,
        ElasticNet=ElasticNet)


def get_regression_model(method):
    model = METHODS[method]()

    return model

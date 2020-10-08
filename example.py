
import numpy as np

from learning import METHODS
from base import Ensemble
from arbitrating import ADE

X = np.random.random(100)
X = X.reshape(20,5)

y = np.random.random(20)

learners = list(METHODS.keys())

model = Ensemble(base_learners=learners)

# fitting the ensemble
model.fit(X, y)

# pruning--keeping 100-(omega*100)% of the best models
model.prune_models()

# predicting - avg output of all models
model.predict(X)

##### arbitrating

####### best lambda is usually between 50 and 100 (depends on data)
####### lambda = 3 is just for example purposes
ade = ADE(base_learners=learners,
          lambda_=3,
          meta_learner="RandomForestRegressor")

ade.fit(X, y)
ade.predict(X)
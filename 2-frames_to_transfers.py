import pandas as pd
import numpy as np
import datetime
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names]

class ToDatetime(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for col in self.attribute_names:
            X[col] = pd.to_datetime(X[col])
        return X

class GetTASQTTC(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y):
        if len(X) != 0:
            ttc = [x.total_seconds() for x in (X.transferred_at - y.submitted_at[0])]
            tasq = [x.total_seconds() for x in (y.submitted_at[0] - X.submitted_at)]
        else:
            ttc = []
            tasq = []
        return tasq, ttc
        

listdir = os.listdir('frames')
listdir.sort()
ys = listdir

cols = ['updated_at', 'started_at', 'transferred_at', 'submitted_at', 'created_at', 'bytes', 'activity']
date_cols = ['updated_at', 'started_at', 'transferred_at', 'submitted_at', 'created_at']
cat_cols = ['activity']

y_pipeline = Pipeline([('selector', DataFrameSelector(cols)),
                       ('datetime', ToDatetime(date_cols))])
x_pipeline = Pipeline([('selector', DataFrameSelector(cols)),
                       ('datetime', ToDatetime(date_cols)),
                       ('ttc', GetTASQTTC())])

y_rep = []
for yname in ys:
    print (yname[:6])
    y = pd.read_csv('frames/'+yname)
    y_prep = y_pipeline.fit_transform(y)
    y_rep.append([y.id.values[0], int(y.bytes.values[0]), y_prep.submitted_at.values[0], 
                  y_prep.started_at.values[0], y_prep.transferred_at.values[0]])
data = np.array(y_rep)
pd.DataFrame(data, columns=['id', 'size', 'submited', 'started', 'ended']).to_hdf('data/transfers.h5', 'table')

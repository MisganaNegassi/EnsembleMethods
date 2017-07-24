#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold


def compute_level1_dataset(X, y, base_models=None):
    if not base_models:
        base_models = [RandomForestClassifier(random_state=1),
                       AdaBoostClassifier(random_state=1, n_estimators=500),
                       DecisionTreeClassifier(random_state=1)]

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25,
                                                        random_state=None)

    base_models_predictions = []
    test_predictions = []

    kf = KFold()
    for i, model in enumerate(base_models):
        for train_index, valid_index in kf.split(train_x):
            model.fit(train_x[train_index], train_y[train_index])
            if i >= len(base_models_predictions):
                base_models_predictions.append(list(model.predict_proba
                                                   (train_x[valid_index])))
            else:
                base_models_predictions[i].extend(list(model.predict_proba
                                                 (train_x[valid_index])))
    for i, base_model in enumerate(base_models):
        base_model.fit(train_x, train_y)
        test_predictions.append(list(base_model.predict_proba(test_x)))

    test_predictions = np.array(test_predictions).transpose()
    base_models_predictions = np.array(base_models_predictions).transpose()

    return base_models_predictions, test_predictions, train_y, test_y

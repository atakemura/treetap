import lightgbm as lgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from rulefit import RuleFit


optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_decision_tree(X, y, random_state=2020):
    early_stopping_dict = {'early_stopping_limit': 30,
                           'early_stop_count': 0,
                           'best_score': None}

    def optuna_early_stopping_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if early_stopping_dict['best_score'] is None:
            early_stopping_dict['best_score'] = study.best_value

        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            if study.best_value > early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        elif study.direction == optuna.study.StudyDirection.MINIMIZE:
            if study.best_value < early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        else:
            raise ValueError('Unknown study direction: {}'.format(study.direction))
        return

    def objective(trial: optuna.Trial):
        # numeric: max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf
        # choice: criterion(gini, entropy)
        params = {'max_depth': trial.suggest_int('max_depth', 2, 10),
                  'min_samples_split': trial.suggest_float('min_samples_split', 0.05, 0.5, step=0.01),
                  'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.05, 0.5, step=0.01),
                  'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5, step=0.01),
                  'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                  }
        dt = DecisionTreeClassifier(**params, random_state=random_state)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_state)
        dt.fit(x_train, y_train)
        y_pred = dt.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        return acc
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def optuna_random_forest(X, y, random_state=2020):
    early_stopping_dict = {'early_stopping_limit': 30,
                           'early_stop_count': 0,
                           'best_score': None}

    def optuna_early_stopping_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if early_stopping_dict['best_score'] is None:
            early_stopping_dict['best_score'] = study.best_value

        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            if study.best_value > early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        elif study.direction == optuna.study.StudyDirection.MINIMIZE:
            if study.best_value < early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        else:
            raise ValueError('Unknown study direction: {}'.format(study.direction))
        return

    def objective(trial: optuna.Trial):
        # numeric: n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf
        # choice: criterion(gini, entropy)
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500, 10),
                  'max_depth': trial.suggest_int('max_depth', 2, 10),
                  'min_samples_split': trial.suggest_float('min_samples_split', 0.05, 0.5, step=0.01),
                  'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.05, 0.5, step=0.01),
                  'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5, step=0.01),
                  'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
                  }
        rf = RandomForestClassifier(**params, random_state=random_state)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_state)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        return acc
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def optuna_lgb(X, y, static_params, random_state=2020):
    early_stopping_dict = {'early_stopping_limit': 30,
                           'early_stop_count': 0,
                           'best_score': None}
    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optuna_early_stopping_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if early_stopping_dict['best_score'] is None:
            early_stopping_dict['best_score'] = study.best_value

        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            if study.best_value > early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        elif study.direction == optuna.study.StudyDirection.MINIMIZE:
            if study.best_value < early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        else:
            raise ValueError('Unknown study direction: {}'.format(study.direction))
        return

    def objective(trial: optuna.Trial):
        params = {'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                  'max_depth': trial.suggest_int('max_depth', 2, 10),
                  'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 500),
                  'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e+1),
                  'feature_fraction': trial.suggest_uniform('feature_fraction', 0.05, 1.0),
                  'subsample': trial.suggest_uniform('subsample', 0.2, 1.0),
                  'subsample_freq': trial.suggest_int('subsample_freq', 1, 20),
                  'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-5, 10),
                  'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-5, 10),
                  }
        all_params = {**params, **static_params}

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, all_params['metric'], valid_name='valid')
        num_boost_round = 1000
        early_stopping = 30
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_state)
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)

        model = lgb.train(all_params, train_data,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping,
                          valid_sets=[valid_data],
                          valid_names=['valid'],
                          callbacks=[pruning_callback],
                          verbose_eval=False)
        if static_params['num_classes'] > 1:
            score = model.best_score['valid']['multi_logloss']
        else:
            score = model.best_score['valid']['binary_logloss']
        return score
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=100, timeout=600, callbacks=[optuna_early_stopping_callback], n_jobs=1)
    return study.best_params


def optuna_rulefit(X, y, rf_params=None, random_state=2020):
    early_stopping_dict = {'early_stopping_limit': 30,
                           'early_stop_count': 0,
                           'best_score': None}

    def optuna_early_stopping_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if early_stopping_dict['best_score'] is None:
            early_stopping_dict['best_score'] = study.best_value

        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            if study.best_value > early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        elif study.direction == optuna.study.StudyDirection.MINIMIZE:
            if study.best_value < early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        else:
            raise ValueError('Unknown study direction: {}'.format(study.direction))
        return

    def objective(trial: optuna.Trial):
        # numeric: tree_size, sample_fract, max_rules, memory_par,
        # bool: lin_standardise, lin_trim_quantile,
        params = {# 'tree_size': trial.suggest_int('tree_size', 50, 500, 10),
                  # 'sample_fract': trial.suggest_float('sample_fract', 0.01, 1.0, step=0.02),
                  # 'max_rules': trial.suggest_int('max_rules', 10, 100),
                  'memory_par': trial.suggest_float('memory_par', 0.0, 1.0, step=0.1),
                  'lin_standardise': trial.suggest_categorical('lin_standardise', [True, False]),
                  'lin_trim_quantile': trial.suggest_categorical('lin_trim_quantile', [True, False]),
        }
        rf = RandomForestClassifier(n_jobs=1, random_state=random_state, **rf_params)
        rfit = RuleFit(tree_generator=rf, max_rules=500, rfmode='classify', n_jobs=1,
                       random_state=random_state, **params)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_state)
        rfit.fit(x_train, y_train, feature_names=x_train.columns)
        try:
            y_pred = rfit.predict(x_valid)
        # this sometimes raises IndexError rulefit.py:281 res_[:,coefs!=0]=res
        except IndexError:
            return 0   # skip this trial
        acc = accuracy_score(y_valid, y_pred)
        return acc
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params

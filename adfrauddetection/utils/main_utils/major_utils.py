import yaml
from adfrauddetection.exception.exception import AdfrauddetectionException
from adfrauddetection.logging.logger import logging
import os,sys
import numpy as np
#import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import xgboost as xgb
from xgboost import XGBClassifier



def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise AdfrauddetectionException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise AdfrauddetectionException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise AdfrauddetectionException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise AdfrauddetectionException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise AdfrauddetectionException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise AdfrauddetectionException(e, sys) from e
    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        classes = np.array([0, 1])
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
        sample_weight = np.where(y_train == 1, class_weight_dict[1], class_weight_dict[0])

        # NEW: stratified CV + PR-AUC (better for imbalance than ROC-AUC)
        from sklearn.model_selection import StratifiedKFold, train_test_split  # NEW
        from sklearn.metrics import average_precision_score                  # NEW
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # NEW
        scoring = "average_precision"  # NEW  (a.k.a. PR-AUC)

        # NEW: small held-out validation (from TRAIN) for XGBoost early stopping
        X_tr, X_val, y_tr, y_val, sw_tr, sw_val = train_test_split(          # NEW
            X_train, y_train, sample_weight, test_size=0.2, stratify=y_train, random_state=42
        )

        # NEW: XGBoost-specific imbalance weight
        neg = int((y_train == 0).sum())  # NEW
        pos = int((y_train == 1).sum())  # NEW
        spw = neg / max(pos, 1)          # NEW

        for i in range(len(list(models))):
            model_key = list(models.keys())[i]
            model = list(models.values())[i]
            para=param.get(model_key, {})

            ### AdaBoost 
            if model_key == "AdaBoost": 
                tree = DecisionTreeClassifier(
                    max_depth=1,                 # NEW: shallow stump works best
                    class_weight=class_weight_dict,
                    random_state=42
                )

                ada = AdaBoostClassifier(
                    estimator=tree,
                    random_state=42
                    )
                grid = GridSearchCV(
                    estimator=ada,
                    param_grid=para,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1
                )
                grid.fit(X_train, y_train, sample_weight=sample_weight)  # NEW
                best = grid.best_estimator_

            ### Gradient Boosting 
            elif model_key == "Gradient Boosting":
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                    scoring=scoring
                )
                grid.fit(X_train, y_train, sample_weight=sample_weight)
                best = grid.best_estimator_ # fitted model with best params
            
            ### XGBoost 
            elif model_key == "XGBoost":
                model.set_params(
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                tree_method="hist",
                scale_pos_weight=spw,   # imbalance handling
                random_state=42
                )

                
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=para,
                    n_iter=min(8, len(para.get("max_depth", [4, 6])) * len(para.get("learning_rate", [0.05, 0.1]))),  # SMALL, FAST
                    cv=cv,                      # CHANGED
                    n_jobs=-1,
                    verbose=1,
                    scoring=scoring             # CHANGED
                )
                search.fit(
                    X_train, y_train,
                    sample_weight=sample_weight                    # NEW
                )
                best = search.best_estimator_ # fitted model with best params
            else:
                continue

            y_score = (best.predict_proba(X_test)[:, 1])   # NEW: fallback

            ap = average_precision_score(y_test, y_score)         # NEW
            report[model_key] = float(ap)                         # CHANGED: store PR-AUC
            models[model_key] = best   
            # Swap in the CV-selected best_estimator_ (already refit on all training data with the best hyperparams),
            # so later `models[best_model_name]` returns the tuned, fitted model—not the original unfitted default.
        return report

    except Exception as e:
        raise AdfrauddetectionException(e, sys)
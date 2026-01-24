import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score
)
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow

# ✅ SAFE dagshub init (never crash pipeline)
try:
    import dagshub
    dagshub.init(
        repo_owner="souravkumar-cloud",
        repo_name="NetworkSecurity",
        mlflow=True,
        quiet=True
    )
except Exception as e:
    logging.warning(f"DagsHub disabled: {e}")


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision_score", classification_metric.precision_score)
            mlflow.log_metric("recall_score", classification_metric.recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"]
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "n_estimators": [8, 16, 32, 64, 128]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128]
            }
        }

        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            params=params
        )

        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # Train metrics
        train_pred = best_model.predict(X_train)
        train_metric = get_classification_score(y_train, train_pred)

        # Test metrics
        test_pred = best_model.predict(X_test)
        test_metric = get_classification_score(y_test, test_pred)

        self.track_mlflow(best_model, train_metric)
        self.track_mlflow(best_model, test_metric)

        # Load preprocessor
        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )

        # ✅ Create NetworkModel instance
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model
        )

        # ✅ Ensure directories exist
        os.makedirs(
            os.path.dirname(self.model_trainer_config.trained_model_file_path),
            exist_ok=True
        )
        os.makedirs("final_model", exist_ok=True)

        # ✅ SAVE CORRECT OBJECTS
        save_object(
            self.model_trainer_config.trained_model_file_path,
            network_model
        )
        save_object(
            "final_model/model.pkl",
            best_model
        )
        save_object(
            "final_model/preprocessor.pkl",
            preprocessor
        )

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

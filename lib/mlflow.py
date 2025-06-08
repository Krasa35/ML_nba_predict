import mlflow
import os
import tempfile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

def basic_log(
    model,
    model_name,
    tags=None,
    params=None,
    metrics=None,
    charts=None,
    input_example=None,
    input_data=None,
    dataset_name=None,
    artifacts=None
):
    """
    Logs model, params, metrics, charts, input data, and additional artifacts to MLflow.
    """
    example = None
    if input_example is None and input_data is not None:
        example = input_data.iloc[:1]
    elif input_example is not None:
        example = input_example
    if example is not None:
        mlflow.sklearn.log_model(model, model_name, input_example=example)
    else:
        mlflow.sklearn.log_model(model, model_name)
    if input_data is not None:
        dataset = mlflow.data.from_pandas(
            input_data,
            name=dataset_name or "input_data",
            # targets=dataset_targets
        )
        mlflow.log_input(dataset, context="training")
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if charts:
        for name, fig in charts.items():
            mlflow.log_figure(fig, name + ".png")
    if artifacts:
        for name, path in artifacts.items():
            mlflow.log_artifact(path, artifact_path=name)
    if tags:
        mlflow.set_tags(tags)

def log_to_mlflow(
    model_or_grid,
    experiment_name,
    model_name,
    tags=dict(),
    params=dict(),
    metrics=dict(),
    charts=dict(),
    artifacts=dict(),
    input_example=None,
    input_data=None,
    dataset_name=None
):
    """
    Logs model training results to MLflow. Supports both GridSearchCV and regular models.
    Also logs the dataset using mlflow.log_input if input_data is provided.

    Args:
        model_or_grid: Fitted model or GridSearchCV object.
        experiment_name (str): MLflow experiment name.
        model_name (str): Name of the model.
        tags (dict, optional): Extra tags.
        params (dict, optional): Params to log (if not GridSearchCV).
        metrics (dict, optional): Metrics to log (if not GridSearchCV).
        charts (dict, optional): Dict of {name: matplotlib_figure} to log as artifacts.
        input_example (array-like, optional): Example input for model signature.
        input_data (array-like or pd.DataFrame, optional): Dataset to log as an artifact and input.
        dataset_name (str, optional): Name for the dataset.
        artifacts (dict, optional): Additional artifacts to log.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)

    # Check if it's a GridSearchCV object
    if hasattr(model_or_grid, "cv_results_") and hasattr(model_or_grid, "best_index_"):
        results = model_or_grid.cv_results_
        params_list = results['params']
        best_idx = model_or_grid.best_index_
        results_df = pd.DataFrame(model_or_grid.cv_results_)
        best_params = params_list[best_idx]
        # Extract best metrics (excluding param columns) and update metrics dict
        best_metrics = {
            metric: float(values[best_idx])
            for metric, values in results.items()
            if isinstance(values[best_idx], (int, float)) and 'param' not in metric
        }
        metrics.update(best_metrics)
        params.update(best_params)
        tags.update({"gridsearch": True})

        with mlflow.start_run(run_name=f"{model_name}_best_model"):
            with tempfile.TemporaryDirectory() as tmpdir:
                cv_results_path = os.path.join(tmpdir, "cv_results.csv")
                results_df.to_csv(cv_results_path, index=False)
                mlflow.log_artifact(cv_results_path, artifact_path="cv_results")
            # Pass artifacts to basic_log
            basic_log(
                model_or_grid.best_estimator_,
                model_name,
                tags,
                params,
                metrics,
                charts,
                input_example,
                input_data,
                dataset_name,
                artifacts
            )
    else:
        with mlflow.start_run(run_name=model_name):
            basic_log(
                model_or_grid,
                model_name,
                tags,
                params,
                metrics,
                charts,
                input_example,
                input_data,
                dataset_name,
                artifacts
            )

def get_experiment_and_model_name(clf, max_score):
    if max_score == 270:
        experiment_name = 'all-nba_TeamPrediction'
    elif max_score == 180:
        experiment_name = 'all-rookie_TeamPrediction'
    else:
        experiment_name = 'data_correlations'

    search = type(clf).__name__
    model = type(clf.best_estimator_).__name__
    model_name = f'{search}_{model}'

    return experiment_name, model_name
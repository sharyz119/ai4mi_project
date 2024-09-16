import joblib
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history

# Load the study
study = joblib.load("optuna_study.pkl")

# Plot parameter importances
fig = plot_param_importances(study)
fig.show()

# Plot optimization history
fig = plot_optimization_history(study)
fig.show()
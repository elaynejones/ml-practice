#define serach space
from azureml.train.hyperdrive import choice, normal

param_space = {
                 '--batch_size': choice(16, 32, 64),
                 '--learning_rate': normal(10, 3)
              }

 #grid sampling example
 from azureml.train.hyperdrive import GridParameterSampling, choice

param_space = {
                 '--batch_size': choice(16, 32, 64),
                 '--learning_rate': choice(0.01, 0.1, 1.0)
              }

param_sampling = GridParameterSampling(param_space)             


#random sampling
from azureml.train.hyperdrive import RandomParameterSampling, choice, normal

param_space = {
                 '--batch_size': choice(16, 32, 64),
                 '--learning_rate': normal(10, 3)
              }

param_sampling = RandomParameterSampling(param_space)


#Bayesian sampling
from azureml.train.hyperdrive import BayesianParameterSampling, choice, uniform

param_space = {
                 '--batch_size': choice(16, 32, 64),
                 '--learning_rate': uniform(0.5, 0.1)
              }

param_sampling = BayesianParameterSampling(param_space)


#bandit policy
from azureml.train.hyperdrive import BanditPolicy

early_termination_policy = BanditPolicy(slack_amount = 0.2,
                                        evaluation_interval=1,
                                        delay_evaluation=5)


#median stopping
from azureml.train.hyperdrive import MedianStoppingPolicy

early_termination_policy = MedianStoppingPolicy(evaluation_interval=1,
                                                delay_evaluation=5)


 #truncation selection
from azureml.train.hyperdrive import TruncationSelectionPolicy

early_termination_policy = TruncationSelectionPolicy(truncation_percentage=10,
                                                     evaluation_interval=1,
                                                     delay_evaluation=5)                                            
                                                     
#tuning script
import argparse
import joblib
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# load the training dataset
data = run.input_datasets['training_data'].to_pandas_dataframe()

# Separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()


#configure & run
from azureml.core import Experiment
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal

# Assumes ws, script_config and param_sampling are already defined

hyperdrive = HyperDriveConfig(run_config=script_config,
                              hyperparameter_sampling=param_sampling,
                              policy=None,
                              primary_metric_name='Accuracy',
                              primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                              max_total_runs=6,
                              max_concurrent_runs=4)

experiment = Experiment(workspace = ws, name = 'hyperdrive_training')
hyperdrive_run = experiment.submit(config=hyperdrive)


#monitor & review runs
for child_run in run.get_children():
    print(child_run.id, child_run.get_metrics())



##EXERCISE

import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


from azureml.core import Dataset

default_ds = ws.get_default_datastore()

if 'diabetes dataset' not in ws.datasets:
    default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'], # Upload the diabetes csv files in /data
                        target_path='diabetes-data/', # Put it in a folder path in the datastore
                        overwrite=True, # Replace existing files of the same name
                        show_progress=True)

    #Create a tabular dataset from the path on the datastore (this may take a short while)
    tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

    # Register the tabular dataset
    try:
        tab_data_set = tab_data_set.register(workspace=ws, 
                                name='diabetes dataset',
                                description='diabetes data',
                                tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    print('Dataset already registered.')



import os

experiment_folder = 'diabetes_training-hyperdrive'
os.makedirs(experiment_folder, exist_ok=True)

print('Folder ready.')


%%writefile $experiment_folder/diabetes_training.py
# Import libraries
import os
import argparse
import joblib
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Set regularization parameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
print("Loading Data...")
diabetes = run.input_datasets['diabetes'].to_pandas_dataframe() # Get the training data from the estimator input


# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))


os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()



from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "your-compute-cluster"

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)


from azureml.core import Experiment
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive import GridParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.widgets import RunDetails


# Sample a range of parameter values
params = GridParameterSampling(
    {
        # There's only one parameter, so grid sampling will try each value - with multiple parameters it would try every combination
        '--regularization': choice(0.001, 0.005, 0.01, 0.05, 0.1, 1.0)
    }
)


# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create an estimator that uses the remote compute
hyper_estimator = SKLearn(source_directory=experiment_folder,
                          inputs=[diabetes_ds.as_named_input('diabetes')], # Pass the dataset as an input...
                          pip_packages=['azureml-sdk'], # ...so we need azureml-dataprep (it's in the SDK!)
                          entry_script='diabetes_training.py',
                          compute_target = training_cluster,)


# Configure hyperdrive settings
hyperdrive = HyperDriveConfig(estimator=hyper_estimator, 
                          hyperparameter_sampling=params, 
                          policy=None, 
                          primary_metric_name='AUC', 
                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                          max_total_runs=6,
                          max_concurrent_runs=4)

# Run the experiment
experiment = Experiment(workspace = ws, name = 'diabates_training_hyperdrive')
run = experiment.submit(config=hyperdrive)

# Show the status in the notebook as the experiment runs
RunDetails(run).show()
run.wait_for_completion()



for child_run in run.get_children_sorted_by_primary_metric():
    print(child_run)

best_run = run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details() ['runDefinition']['arguments']

print('Best Run Id: ', best_run.id)
print(' -AUC:', best_run_metrics['AUC'])
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Regularization Rate:',parameter_values)



from azureml.core import Model

# Register model
best_run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                        tags={'Training context':'Hyperdrive'},
                        properties={'AUC': best_run_metrics['AUC'], 'Accuracy': best_run_metrics['Accuracy']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')
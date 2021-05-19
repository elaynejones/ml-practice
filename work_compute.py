#make enviro from specification file
name: py_env
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - pip:
    - azureml-defaults

#then use this to make ML enviro
from azureml.core import Environment

env = Environment.from_conda_specification(name='training_environment',
                                           file_path='./conda.yml')


#make enviro from existing Conda
from azureml.core import Environment

env = Environment.from_existing_conda_environment(name='training_environment',
                                                  conda_environment_name='py_env')


#by specifying pkg
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment('training_environment')
deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas','numpy'],
                                pip_packages=['azureml-defaults'])
env.python.conda_dependencies = deps


#configuring enviro containers
env.docker.enabled = True
deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas','pip'],                      
                                pip_packages=['azureml-defaults']
env.python.conda_dependencies = deps


#override w custom img
env.docker.base_image='my-base-image'
env.docker.base_image_registry='myregistry.azurecr.io/myimage'


#override how AZ auto handles pkg dependencies etc
env.python.user_managed_dependencies=True
env.python.interpreter_path = '/opt/miniconda/bin/python'


#register
env.register(workspace=ws)

#view registered
from azureml.core import Environment

env_names = Environment.list(workspace=ws)
for env_name in env_names:
    print('Name:',env_name)

#retrieve enviro
from azureml.core import Environment
from azureml.train.estimator import Estimator

training_env = Environment.get(workspace=ws, name='training_environment')
estimator = Estimator(source_directory='experiment_folder'
                      entry_script='training_script.py',
                      compute_target='local',
                      environment_definition=training_env)


 #make Managed COmpute target
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'aml-cluster'

# Define compute configuration
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                       min_nodes=0, max_nodes=4,
                                                       vm_priority='dedicated')

# Create the compute
aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
aml_cluster.wait_for_completion(show_output=True)      



#unmanaged Compute Target
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, DatabricksCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'db_cluster'

# Define configuration for existing Azure Databricks cluster
db_workspace_name = 'db_workspace'
db_resource_group = 'db_resource_group'
db_access_token = '1234-abc-5678-defg-90...'
db_config = DatabricksCompute.attach_configuration(resource_group=db_resource_group,
                                                   workspace_name=db_workspace_name,
                                                   access_token=db_access_token)

# Create the compute
databricks_compute = ComputeTarget.attach(ws, compute_name, db_config)
databricks_compute.wait_for_completion(True)


#check for existing compute
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = "aml-cluster"

# Check if the compute target exists
try:
    aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing cluster.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                           max_nodes=4)
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)



#use compute target by config estimator
from azureml.core import Environment, ScriptRunConfig

compute_name = 'aml-cluster'

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                environment=env,
                                compute_target=compute_name)



#specify ComputeTarget
from azureml.core import Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

compute_name = "aml-cluster"

training_cluster = ComputeTarget(workspace=ws, name=compute_name)

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                environment=env,
                                compute_target=training_cluster)



 #EXERCISE
 # 
 # 
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


#make training script

import os

# Create a folder for the experiment files
experiment_folder = 'diabetes_training_logistic'
os.makedirs(experiment_folder, exist_ok=True)
print(experiment_folder, 'folder created')


%%writefile $experiment_folder/diabetes_training.py
# Import libraries
import os
import argparse
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Set regularization hyperparameter (passed as an argument to the script)
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()


# load the diabetes data (passed as an input dataset)
print("Loading Data...")
diabetes = run.input_datasets['diabetes'].to_pandas_dataframe()

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


#define enviro
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create a Python environment for the experiment
diabetes_env = Environment("diabetes-experiment-env")
diabetes_env.python.user_managed_dependencies = False # Let Azure ML manage dependencies
diabetes_env.docker.enabled = True # Use a docker container

# Create a set of package dependencies (conda or pip as required)
diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn'],
                                          pip_packages=['azureml-defaults', 'azureml-dataprep[pandas]'])

# Add the dependencies to the environment
diabetes_env.python.conda_dependencies = diabetes_packages

print(diabetes_env.name, 'defined.')


#assign enviro to estimator & submit to exp
from azureml.train.estimator import Estimator
from azureml.core import Experiment
from azureml.widgets import RunDetails

# Set the script parameters
script_params = {
    '--regularization': 0.1
}

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create an estimator
estimator = Estimator(source_directory=experiment_folder,
                      inputs=[diabetes_ds.as_named_input('diabetes')],
                      script_params=script_params,
                      compute_target = 'local',
                      environment_definition = diabetes_env,
                      entry_script='diabetes_training.py')

# Create an experiment
experiment = Experiment(workspace = ws, name = 'diabetes-training')

# Run the experiment
run = experiment.submit(config=estimator)
# Show the run details while running
RunDetails(run).show()
run.wait_for_completion()

# Register the environment
diabetes_env.register(workspace=ws)


#run on remote compute

#check for existing

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

from azureml.train.estimator import Estimator
from azureml.core import Environment, Experiment
from azureml.widgets import RunDetails

# Get the environment
registered_env = Environment.get(ws, 'diabetes-experiment-env')

# Set the script parameters
script_params = {
    '--regularization': 0.1
}

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create an estimator
estimator = Estimator(source_directory=experiment_folder,
                      inputs=[diabetes_ds.as_named_input('diabetes')],
                      script_params=script_params,
                      compute_target = cluster_name, # Run the experiment on the remote compute target
                      environment_definition = registered_env,
                      entry_script='diabetes_training.py')

# Create an experiment
experiment = Experiment(workspace = ws, name = 'diabetes-training')

# Run the experiment
run = experiment.submit(config=estimator)
# Show the run details while running
RunDetails(run).show()
run.wait_for_completion()


# Get logged metrics
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)

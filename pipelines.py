#define 2 steps to prepare data, train model
from azureml.pipeline.steps import PythonScriptStep

# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster')

# Step to train a model
step2 = PythonScriptStep(name = 'train model',
                         source_directory = 'scripts',
                         script_name = 'train_model.py',
                         compute_target = 'aml-cluster')

from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

# Construct the pipeline
train_pipeline = Pipeline(workspace = ws, steps = [step1,step2])

# Create an experiment and run the pipeline
experiment = Experiment(workspace = ws, name = 'training-pipeline')
pipeline_run = experiment.submit(train_pipeline)


#Pass Data bn Pipeline steps w PipelineData obj
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# Get a dataset for the initial data
raw_ds = Dataset.get_by_name(ws, 'raw_dataset')

# Define a PipelineData object to pass data between steps
data_store = ws.get_default_datastore()
prepped_data = PipelineData('prepped',  datastore=data_store)

# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         # Script arguments include PipelineData
                         arguments = ['--raw-ds', raw_ds.as_named_input('raw_data'),
                                      '--out_folder', prepped_data],
                         # Specify PipelineData as output
                         outputs=[prepped_data])

# Step to run an estimator
step2 = PythonScriptStep(name = 'train model',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         # Pass as script argument
                         arguments=['--in_folder', prepped_data],
                         # Specify PipelineData as input
                         inputs=[prepped_data])



#can use pipelinedata like local folder
# code in data_prep.py
from azureml.core import Run
import argparse
import os

# Get the experiment run context
run = Run.get_context()

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--raw-ds', type=str, dest='raw_dataset_id')
parser.add_argument('--out_folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

# Get input dataset as dataframe
raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()

# code to prep data (in this case, just select specific columns)
prepped_df = raw_df[['col1', 'col2', 'col3']]

# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'prepped_data.csv')
prepped_df.to_csv(output_path)



#Reusing Steps
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         runconfig = run_config,
                         inputs=[raw_ds.as_named_input('raw_data')],
                         outputs=[prepped_data],
                         arguments = ['--folder', prepped_data]),
                         # Disable step reuse
                         allow_reuse = False)


#force all steps to run regardless of indiv config
pipeline_run = experiment.submit(train_pipeline, regenerate_outputs=True)


#publish pipeline
published_pipeline = pipeline.publish(name='training_pipeline',
                                          description='Model training pipeline',
                                          version='1.0')


#can also publish like this
# Get the most recent run of the pipeline
pipeline_experiment = ws.experiments.get('training-pipeline')
run = list(pipeline_experiment.get_runs())[0]

# Publish the pipeline from the run
published_pipeline = run.publish_pipeline(name='training_pipeline',
                                          description='Model training pipeline',
                                          version='1.0')


#find URL of endpoint
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)     



#use the published pipeline
import requests

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline"})
run_id = response.json()["Id"]
print(run_id)


#example parameter code for regularization rate
from azureml.pipeline.core.graph import PipelineParameter

reg_param = PipelineParameter(name='reg_rate', default_value=0.01)

...

step2 = PythonScriptStep(name = 'train model',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         # Pass parameter as script argument
                         arguments=['--in_folder', prepped_data,
                                    '--reg', reg_param],
                         inputs=[prepped_data])



#run pipeline w parameter
response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline",
                               "ParameterAssignments": {"reg_rate": 0.1}})


#set schedule
from azureml.pipeline.core import ScheduleRecurrence, Schedule

daily = ScheduleRecurrence(frequency='Day', interval=1)
pipeline_schedule = Schedule.create(ws, name='Daily Training',
                                        description='trains model every day',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='Training_Pipeline',
                                        recurrence=daily)                              

#schedule it to run when data changes
from azureml.core import Datastore
from azureml.pipeline.core import Schedule

training_datastore = Datastore(workspace=ws, name='blob_data')
pipeline_schedule = Schedule.create(ws, name='Reactive Training',
                                    description='trains model on data change',
                                    pipeline_id=published_pipeline_id,
                                    experiment_name='Training_Pipeline',
                                    datastore=training_datastore,
                                    path_on_datastore='data/training')




#EXERCISE


import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

#prepare training data
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



#pipeline steps
import os
# Create a folder for the pipeline step files
experiment_folder = 'diabetes_pipeline'
os.makedirs(experiment_folder, exist_ok=True)

print(experiment_folder)


%%writefile $experiment_folder/train_diabetes.py
# Import libraries
from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, dest='output_folder', default="diabetes_model", help='output folder')
args = parser.parse_args()
output_folder = args.output_folder

# Get the experiment run context
run = Run.get_context()

# load the diabetes data (passed as an input dataset)
print("Loading Data...")
diabetes = run.input_datasets['diabetes_train'].to_pandas_dataframe()

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values


# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train adecision tree model
print('Training a decision tree model')
model = DecisionTreeClassifier().fit(X_train, y_train)

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

# Save the trained model
os.makedirs(output_folder, exist_ok=True)
output_path = output_folder + "/model.pkl"
joblib.dump(value=model, filename=output_path)

run.complete()


#register model
%%writefile $experiment_folder/register_diabetes.py
# Import libraries
import argparse
import joblib
from azureml.core import Workspace, Model, Run

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, dest='model_folder', default="diabetes_model", help='model location')
args = parser.parse_args()
model_folder = args.model_folder

# Get the experiment run context
run = Run.get_context()

# load the model
print("Loading model from " + model_folder)
model_file = model_folder + "/model.pkl"
model = joblib.load(model_file)

Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'diabetes_model',
               tags={'Training context':'Pipeline'})

run.complete()


#make compute enviro
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "your-compute-cluster"

try:
    # Check for existing compute target
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)


from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration

# Create a Python environment for the experiment
diabetes_env = Environment("diabetes-pipeline-env")
diabetes_env.python.user_managed_dependencies = False # Let Azure ML manage dependencies
diabetes_env.docker.enabled = True # Use a docker container

# Create a set of package dependencies
diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn','pandas'],
                                             pip_packages=['azureml-defaults','azureml-dataprep[pandas]'])

# Add the dependencies to the environment
diabetes_env.python.conda_dependencies = diabetes_packages

# Register the environment (just in case you want to use it again)
diabetes_env.register(workspace=ws)
registered_env = Environment.get(ws, 'diabetes-pipeline-env')

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = pipeline_cluster

# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

print ("Run configuration created.")       



#create and run pipeline
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.train.estimator import Estimator

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create a PipelineData (temporary Data Reference) for the model folder
model_folder = PipelineData("model_folder", datastore=ws.get_default_datastore())

estimator = Estimator(source_directory=experiment_folder,
                        compute_target = pipeline_cluster,
                        environment_definition=pipeline_run_config.environment,
                        entry_script='train_diabetes.py')

# Step 1, run the estimator to train the model
train_step = EstimatorStep(name = "Train Model",
                           estimator=estimator, 
                           estimator_entry_script_arguments=['--output_folder', model_folder],
                           inputs=[diabetes_ds.as_named_input('diabetes_train')],
                           outputs=[model_folder],
                           compute_target = pipeline_cluster,
                           allow_reuse = True)


 # Step 2, run the model registration script
register_step = PythonScriptStep(name = "Register Model",
                                source_directory = experiment_folder,
                                script_name = "register_diabetes.py",
                                arguments = ['--model_folder', model_folder],
                                inputs=[model_folder],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

print("Pipeline steps defined")  


from azureml.core import Experiment
from azureml.pipeline.core import Pipeline

# Construct the pipeline
pipeline_steps = [train_step, register_step]
pipeline = Pipeline(workspace = ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create an experiment and run the pipeline
experiment = Experiment(workspace = ws, name = 'diabetes-training-pipeline')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")
pipeline_run.wait_for_completion(show_output=True)


#verify new model was made

from azureml.core import Model

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')


#publish pipeline
published_pipeline = pipeline.publish(name="Diabetes_Training_Pipeline",
                                      description="Trains diabetes model",
                                      version="1.0")
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)

#use authorization header for client apps
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()

#track the pipeline exp while it runs

import requests
experiment_name = 'Run-diabetes-pipeline'

response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": experiment_name})
run_id = response.json()["Id"]
run_id


from azureml.pipeline.core.run import PipelineRun

published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
published_pipeline_run.wait_for_completion(show_output=True)

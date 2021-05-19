from azureml.core import Workspace, Datastore

ws = Workspace.from_config()

# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(workspace=ws, 
                                                  datastore_name='blob_data', 
                                                  container_name='data_container',
                                                  account_name='az_store_acct',
                                                  account_key='123456abcde789…')
#Manage them
for ds_name in ws.datastores:
    print(ds_name)

blob_store = Datastore.get(ws, datastore_name='blob_data')

default_store = ws.get_default_datastore()

ws.set_default_datastore('blob_data')


#create tabular dataset
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
csv_paths = [(blob_ds, 'data/files/current_data.csv'),
             (blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')

#make file dataset
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')


#retrieve registered
import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['csv_table']

# Get a dataset by name from the datasets class
ds2 = Dataset.get_by_name(ws, 'img_files')


#versioning
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
             (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)


#retrieve
img_ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)


#work w tabular
df = tab_ds.to_pandas_dataframe()
# code to work with dataframe goes here, for example:
print(df.head())


#use script arg
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds],
                                environment=env)

#Script
from azureml.core import Run, Dataset

parser.add_argument('--ds', type=str, dest='dataset_id')
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace
dataset = Dataset.get_by_id(ws, id=args.dataset_id)
data = dataset.to_pandas_dataframe()



#use named input
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds.as_named_input('my_dataset')],
                                environment=env)


#script
from azureml.core import Run

parser.add_argument('--ds', type=str, dest='ds_id')
args = parser.parse_args()

run = Run.get_context()
dataset = run.input_datasets['my_dataset']
data = dataset.to_pandas_dataframe()


#work w files
for file_path in file_ds.to_path():
    print(file_path)

#script Arg
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', file_ds.as_download()],
                                environment=env)

#script
from azureml.core import Run
import glob

parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

imgs = glob.glob(ds_ref + "/*.jpg")


#named input
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', file_ds.as_named_input('my_ds').as_download()],
                                environment=env)
#script
from azureml.core import Run
import glob

parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

dataset = run.input_datasets['my_ds']
imgs= glob.glob(dataset + "/*.jpg")



#EXERCISE

import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get the default datastore
default_ds = ws.get_default_datastore()

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:
    print(ds_name, "- Default =", ds_name == default_ds.name)



default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'], # Upload the diabetes csv files in /data
                       target_path='diabetes-data/', # Put it in a folder path in the datastore
                       overwrite=True, # Replace existing files of the same name
                       show_progress=True)

data_ref = default_ds.path('diabetes-data').as_download(path_on_compute='diabetes_data')
print(data_ref)



import os

# Create a folder for the experiment files
experiment_folder = 'diabetes_training_from_datastore'
os.makedirs(experiment_folder, exist_ok=True)
print(experiment_folder, 'folder created.')

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
from sklearn.metrics import roc_curve

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder reference')
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# load the diabetes data from the data reference
data_folder = args.data_folder
print("Loading data from", data_folder)
# Load all files and concatenate their contents as a single dataframe
all_files = os.listdir(data_folder)
diabetes = pd.concat((pd.read_csv(os.path.join(data_folder,csv_file)) for csv_file in all_files))


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



from azureml.train.sklearn import SKLearn
from azureml.core import Experiment
from azureml.widgets import RunDetails

# Set up the parameters
script_params = {
    '--regularization': 0.1, # regularization rate
    '--data-folder': data_ref # data reference to download files from datastore
}


# Create an estimator
estimator = SKLearn(source_directory=experiment_folder,
                    entry_script='diabetes_training.py',
                    script_params=script_params,
                    compute_target = 'local'
                   )

# Create an experiment
experiment_name = 'diabetes-training'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment
run = experiment.submit(config=estimator)
# Show the run details while running
RunDetails(run).show()
run.wait_for_completion()




#work w tabular
from azureml.core import Dataset

# Get the default datastore
default_ds = ws.get_default_datastore()

#Create a tabular dataset from the path on the datastore (this may take a short while)
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

# Display the first 20 rows as a Pandas dataframe
tab_data_set.take(20).to_pandas_dataframe()





#Create a file dataset from the path on the datastore (this may take a short while)
file_data_set = Dataset.File.from_files(path=(default_ds, 'diabetes-data/*.csv'))

# Get the files in the dataset
for file_path in file_data_set.to_path():
    print(file_path)



# Register the tabular dataset
try:
    tab_data_set = tab_data_set.register(workspace=ws, 
                                        name='diabetes dataset',
                                        description='diabetes data',
                                        tags = {'format':'CSV'},
                                        create_new_version=True)
except Exception as ex:
    print(ex)

# Register the file dataset
try:
    file_data_set = file_data_set.register(workspace=ws,
                                            name='diabetes file dataset',
                                            description='diabetes files',
                                            tags = {'format':'CSV'},
                                            create_new_version=True)
except Exception as ex:
    print(ex)

print('Datasets registered')


#get list of datasets
print("Datasets:")
for dataset_name in list(ws.datasets.keys()):
    dataset = Dataset.get_by_name(ws, dataset_name)
    print("\t", dataset.name, 'version', dataset.version)


#test w tabular
import os

# Create a folder for the experiment files
experiment_folder = 'diabetes_training_from_tab_dataset'
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
from sklearn.metrics import roc_curve

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


#make estimator to run script, defined named input for train DS


from azureml.train.sklearn import SKLearn
from azureml.core import Experiment
from azureml.widgets import RunDetails

# Set the script parameters
script_params = {
    '--regularization': 0.1
}

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create an estimator
estimator = SKLearn(source_directory=experiment_folder,
                    entry_script='diabetes_training.py',
                    script_params=script_params,
                    compute_target = 'local',
                    inputs=[diabetes_ds.as_named_input('diabetes')], # Pass the Dataset object as an input...
                    pip_packages=['azureml-dataprep[pandas]'] # ...so you need the dataprep package
                   )

# Create an experiment
experiment_name = 'diabetes-training'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment
run = experiment.submit(config=estimator)
# Show the run details while running
RunDetails(run).show()
run.wait_for_completion()



#test exp from file 


import os

# Create a folder for the experiment files
experiment_folder = 'diabetes_training_from_file_dataset'
os.makedirs(experiment_folder, exist_ok=True)
print(experiment_folder, 'folder created')


%%writefile $experiment_folder/diabetes_training.py
# Import libraries
import os
import argparse
from azureml.core import Workspace, Dataset, Experiment, Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import glob

# Set regularization hyperparameter (passed as an argument to the script)
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()


# load the diabetes dataset
print("Loading Data...")
data_path = run.input_datasets['diabetes'] # Get the training data from the estimator input
all_files = glob.glob(data_path + "/*.csv")
diabetes = pd.concat((pd.read_csv(f) for f in all_files))

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




from azureml.train.sklearn import SKLearn
from azureml.core import Experiment
from azureml.widgets import RunDetails

# Set the script parameters
script_params = {
    '--regularization': 0.1
}

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes file dataset")

# Create an estimator
estimator = SKLearn(source_directory=experiment_folder,
                    entry_script='diabetes_training.py',
                    script_params=script_params,
                    compute_target = 'local',
                    inputs=[diabetes_ds.as_named_input('diabetes').as_download(path_on_compute='diabetes_data')], # Pass the Dataset object as an input
                    pip_packages=['azureml-dataprep[pandas]'] # so we need the dataprep package
                   )

# Create an experiment
experiment_name = 'diabetes-training'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment
run = experiment.submit(config=estimator)
# Show the run details while running
RunDetails(run).show()
run.wait_for_completion()

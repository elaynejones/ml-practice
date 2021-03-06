from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Prepare the dataset
diabetes = pd.read_csv('data.csv')
X, y = data[['Feature1','Feature2','Feature3']].values, data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()


from azureml.train.estimator import Estimator
from azureml.core import Experiment

# Create an estimator
estimator = Estimator(source_directory='experiment_folder',
                      entry_script='training_script.py',
                      compute_target='local',
                      conda_packages=['scikit-learn']
                      )

# Create and run an experiment
experiment = Experiment(workspace = ws, name = 'training_experiment')
run = experiment.submit(config=estimator)


from azureml.train.sklearn import SKLearn
from azureml.core import Experiment

# Create an estimator
estimator = SKLearn(source_directory='experiment_folder',
                    entry_script='training_script.py'
                    compute_target='local'
                    )

# Create and run an experiment
experiment = Experiment(workspace = ws, name = 'training_experiment')
run = experiment.submit(config=estimator)


from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg

# Prepare the dataset
diabetes = pd.read_csv('data.csv')
X, y = data[['Feature1','Feature2','Feature3']].values, data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()


from azureml.train.sklearn import SKLearn
from azureml.core import Experiment

# Create an estimator
estimator = SKLearn(source_directory='experiment_folder',
                    entry_script='training_script.py',
                    script_params = {'--reg_rate': 0.1},
                    compute_target='local'
                    )

# Create and run an experiment
experiment = Experiment(workspace = ws, name = 'training_experiment')
run = experiment.submit(config=estimator)


# "run" is a reference to a completed experiment run

# List the files generated by the experiment
for file in run.get_file_names():
    print(file)

# Download a named file
run.download_file(name='outputs/model.pkl', output_file_path='model.pkl')


from azureml.core import Model

model = Model.register(workspace=ws,
                       model_name='classification_model',
                       model_path='model.pkl', # local path
                       description='A classification model',
                       tags={'dept': 'sales'},
                       model_framework=Model.Framework.SCIKITLEARN,
                       model_framework_version='0.20.3')

#alternatively
run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl', # run outputs path
                    description='A classification model',
                    tags={'dept': 'sales'},
                    model_framework=Model.Framework.SCIKITLEARN,
                    model_framework_version='0.20.3')


#view registered models
from azureml.core import Model

for model in Model.list(ws):
    # Get model name and auto-generated version
    print(model.name, 'version:', model.version)


#NOW TRAIN MODEL

#1 connect to workspace
import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


import os, shutil

# Create a folder for the experiment files
training_folder = 'diabetes-training'
os.makedirs(training_folder, exist_ok=True)

# Copy the data file into the experiment folder
shutil.copy('data/diabetes.csv', os.path.join(training_folder, "diabetes.csv"))

%%writefile $training_folder/diabetes_training.py
# Import libraries
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv('diabetes.csv')

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Set regularization hyperparameter
reg = 0.01

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

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()


from azureml.train.estimator import Estimator
from azureml.core import Experiment

# Create an estimator
estimator = Estimator(source_directory=training_folder,
                      entry_script='diabetes_training.py',
                      compute_target='local',
                      conda_packages=['scikit-learn']
                      )

# Create an experiment
experiment_name = 'diabetes-training'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment based on the estimator
run = experiment.submit(config=estimator)
run.wait_for_completion(show_output=True)

from azureml.widgets import RunDetails

RunDetails(run).show()


# Get logged metrics
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)


#register trained model

from azureml.core import Model

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Estimator'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

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


import os, shutil

# Create a folder for the experiment files
training_folder = 'diabetes-training-params'
os.makedirs(training_folder, exist_ok=True)

# Copy the data file into the experiment folder
shutil.copy('data/diabetes.csv', os.path.join(training_folder, "diabetes.csv"))


%%writefile $training_folder/diabetes_training.py
# Import libraries
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg

# load the diabetes dataset
print("Loading Data...")
# load the diabetes dataset
diabetes = pd.read_csv('diabetes.csv')


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
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()


#SK LEarn specific estimator
from azureml.train.sklearn import SKLearn
from azureml.widgets import RunDetails

# Create an estimator
estimator = SKLearn(source_directory=training_folder,
                    entry_script='diabetes_training.py',
                    script_params = {'--reg_rate': 0.1},
                    compute_target='local'
                    )

# Create an experiment
experiment_name = 'diabetes-training'
experiment = Experiment(workspace = ws, name = experiment_name)

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

#Register New version
from azureml.core import Model

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Parameterized SKLearn Estimator'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

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
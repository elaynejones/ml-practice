#step 1 - register
from azureml.core import Model

classification_model = Model.register(workspace=your_workspace,
                                      model_name='classification_model',
                                      model_path='model.pkl', # local path
                                      description='A classification model')


# step 2 - scoring script
import os
import numpy as np
from azureml.core import Model
import joblib

def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)

def run(mini_batch):
    # This runs for each batch
    resultList = []

    # process each file in the batch
    for f in mini_batch:
        # Read comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-dimensional array for model input
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList



#step 3 - make pipeline w ParallelRunStep
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import Pipeline

# Get the batch dataset for input
batch_data_set = ws.datasets['batch-data']

# Set the output location
default_ds = ws.get_default_datastore()
output_dir = PipelineData(name='inferences',
                          datastore=default_ds,
                          output_path_on_compute='results')

# Define the parallel run step step configuration
parallel_run_config = ParallelRunConfig(
    source_directory='batch_scripts',
    entry_script="batch_scoring_script.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=batch_env,
    compute_target=aml_cluster,
    node_count=4)

# Create the parallel run step
parallelrun_step = ParallelRunStep(
    name='batch-score',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('batch_data')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)
# Create the pipeline
pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])


#step 4 - run pipeline and get step output
from azureml.core import Experiment

# Run the pipeline as an experiment
pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# Get the outputs from the first (and only) step
prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
prediction_output.download(local_path='results')

# Find the parallel_run_step.txt file
for root, dirs, files in os.walk('results'):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# Load and display the results
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]
print(df)


#step 5 - publish 

#REST service
published_pipeline = pipeline_run.publish_pipeline(name='Batch_Prediction_Pipeline',
                                                   description='Batch pipeline',
                                                   version='1.0')
rest_endpoint = published_pipeline.endpoint

#use endpoint to start job
import requests

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "Batch_Prediction"})
run_id = response.json()["Id"]


#have it run auto
from azureml.pipeline.core import ScheduleRecurrence, Schedule

weekly = ScheduleRecurrence(frequency='Week', interval=1)
pipeline_schedule = Schedule.create(ws, name='Weekly Predictions',
                                        description='batch inferencing',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='Batch_Prediction',
                                        recurrence=weekly)



##EXERCISE


#1 - connect to ws
import azureml.core
from azureml.core import Workspace 

#load from config file
ws = workspace.from_config()
print('ready to use azure {} to work with {}'.format(azureml.core.VERSION, ws.name))


#2 - train / register

from azureml.core import experiment
from azureml.core import model
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#make AZML exp 
experiment = Experiment(workspace=ws, name='diabetes-training')
run = experiment.start_logging()
print('Starting exp:' experiment.name)

#load diabetes dataset
print('Loading Data...')
diabetes = pd.read_csv('data/diabetes.csv')

#sep features / labels
x, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

#split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#train decision tree 
print('training model')
model = DecisionTreeClassifier().fit(x_train, y_train)

#calculate accuracy
y_hat=model.predict(x_test)
acc=np.average(y_hat ==y_test)
print('Accuracy:' acc)
run.log('Accuracy', np.float(acc))

#calc AUC
y_scores = model.predict_proba(x_test)
auc = roc_auc_score(y_test, y_scores[:,1])
print('AUC' + str(auc))
run.log('AUC' , np.float(auc))

#save trained model
model_file = 'diabetes_model.pkl'
joblib.dump(value=model, filename=model_file)
run.upload_file(name= 'outputs/' + model_file, path_or_stream = './' + model_file)


#complete run
run.complete()

#register model
run.register_model(model_path= 'outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Inline Training'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

print('Model trained and registered.')


from azureml.core import Datastore, Dataset
import pandas as pd
import os

#load diabetes data
diabetes = pd.read_csv('data/diabetes2.csv')
#get 100 item sample of fetaure columns
sample=diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].sample(n=100).values

#make folder
batch_folder = './batch-data'
os.makedirs(batch_folder, exist_ok=True)
print('Folder created')

#save each sample as sep file
print('saving files...')
for i in range(100):
    fname=str(i+1) + '.csv'
    sample[i].tofile(os.path.join(batch_folder, fname), sep=', ')
    print('Files saved!')

#upload files to default datastore
print("Uploading files to datastore...")
default_ds = ws.get_default_datastore()
default_ds.upload(src_dir="batch-data", target_path="batch-data", overwrite=True, show_progress=True)

#register dataset for input
batch_data_set=Dataset.file.from_files(path=(default_ds, 'batch-data/', validate = false))
try:
    batch_data_set = batch_data_set.register(workspace=ws, name='batch-data', description='batch data', create_new_version=True)
    except Exception as ex:
        print(ex)

print('done')



from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "your-compute-cluster"


try:
    #check for existing compute
    inference_cluster= ComputeTarget(Workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        inference_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        inference_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)


 #now make pipeline for batch inf
 import os
 #make folder for exp files
 experiment_folder='batch_pipeline'
 os.makedirs(experiment_folder, exist_ok=true)       
 print(experiment_folder)


 #make python script to do the actual work
 import os
 import numpy as np
 from azureml.core import model
 import joblib

 def init():
     global model

     #load model
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)

def run(mini_batch):
    resultlist=[]

    #process each file in batch
    for f in mini_batch:
        #read comma delimited data into an array
        data=np.genfromtxt(f, delimiter=',')
        #reshape into 2D array for prediction (model expects multiple items)
        prediction=model.predict(data.reshape(1,-1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList


from azureml.core import Environment
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.runconfig import CondaDependencies

# Add dependencies required by the model
# For scikit-learn models, you need scikit-learn
# For parallel pipeline steps, you need azureml-core and azureml-dataprep[fuse]
cd = CondaDependencies.create(pip_packages=['scikit-learn','azureml-defaults','azureml-core','azureml-dataprep[fuse]'])

batch_env = Environment(name='batch_environment')
batch_env.python.conda_dependencies = cd
batch_env.docker.enabled = True
batch_env.docker.base_image = DEFAULT_CPU_IMAGE
print('Configuration ready.')

#parallel run - run batch prediction script, gen predictions from input, save results all at same time
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.pipeline.core import PipelineData

default_ds = ws.get_default_datastore()

output_dir = PipelineData(name='inferences', 
                          datastore=default_ds, 
                          output_path_on_compute='diabetes/results')

parallel_run_config = ParallelRunConfig(
    source_directory=experiment_folder,
    entry_script="batch_diabetes.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=batch_env,
    compute_target=inference_cluster,
    node_count=2)

parallelrun_step = ParallelRunStep(
    name='batch-score-diabetes',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('diabetes_batch')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)

print('Steps defined')

#now put into pipeline
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline

pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
print('Running pipeline...')
pipeline_run.wait_for_completion(show_output=True)


#get resulting predictions

import pandas as pd
import shutil

shutil.rmtree('diabetes-results', ignore_errors=True)

prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
prediction_output.download(local_path='diabetes-results')


for root, dirs, files in os.walk('diabetes-results'):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# cleanup output format
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]

# Display the first 20 results
df.head(20)


#publish pipeline

published_pipeline = pipeline_run.publish_pipeline(
    name='Diabetes_Parallel_Batch_Pipeline', description='Batch scoring of diabetes data', version='1.0')

published_pipeline

rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)


from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print('Authentication header ready.')


import requests

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "Batch_Pipeline_via_REST"})
run_id = response.json()["Id"]
run_id


from azureml.pipeline.core.run import PipelineRun

published_pipeline_run = PipelineRun(ws.experiments["Batch_Pipeline_via_REST"], run_id)
print('Running pipeline...')
published_pipeline_run.wait_for_completion(show_output=True)


import pandas as pd
import shutil

shutil.rmtree("diabetes-results", ignore_errors=True)

prediction_run = next(published_pipeline_run.get_children())
prediction_output = prediction_run.get_output_data("inferences")
prediction_output.download(local_path="diabetes-results")


for root, dirs, files in os.walk("diabetes-results"):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# cleanup output format
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]

# Display the first 20 results
df.head(20)
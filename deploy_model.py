#register trained model
from azureml.core import Model

classification_model = Model.register(workspace=ws,
                       model_name='classification_model',
                       model_path='model.pkl', # local path
                       description='A classification model')

#could also use run
run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl', # run outputs path
                    description='A classification model')


#define init config
import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return predictions.tolist()


#make enviro
from azureml.core.conda_dependencies import CondaDependencies

# Add the dependencies for your model
myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

# Save the environment config as a .yml file
env_file = 'service_files/env.yml'
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)


#combine entry and enviro into InferenceConfig
from azureml.core.model import InferenceConfig

classifier_inference_config = InferenceConfig(runtime= "python",
                                              source_directory = 'service_files',
                                              entry_script="score.py",
                                              conda_file="env.yml")

#make AKS cluster
from azureml.core.compute import ComputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)     


#define deploy config
from azureml.core.webservice import AksWebservice

classifier_deploy_config = AksWebservice.deploy_configuration(cpu_cores = 1,
                                                              memory_gb = 1)


#deploy model
from azureml.core.model import Model

model = ws.models['classification_model']
service = Model.deploy(workspace=ws,
                       name = 'classifier-service',
                       models = [model],
                       inference_config = classifier_inference_config,
                       deployment_config = classifier_deploy_config,
                       deployment_target = production_cluster)
service.wait_for_deployment(show_output = True)  


#use SDK to consume model
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Call the web service, passing the input data
response = service.run(input_data = json_data)

# Get the predictions
predictions = json.loads(response)

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )


#use REST endpoint
import requests
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Set the content type in the request headers
request_headers = { 'Content-Type':'application/json' }

# Call the service
response = requests.post(url = endpoint,
                         data = json_data,
                         headers = request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )



#get keys
primary_key, secondary_key = service.get_keys()   


#make call to REST endpoint
import requests
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Set the content type in the request headers
request_headers = { "Content-Type":"application/json",
                    "Authorization":"Bearer " + key_or_token }

# Call the service
response = requests.post(url = endpoint,
                         data = json_data,
                         headers = request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )


#check service state
from azureml.core.webservice import AksWebservice

# Get the deployed service
service = AksWebservice(name='classifier-service', workspace=ws)

# Check its state
print(service.state)   

#review service logs
print(service.get_logs())


#deploy to local container to diagnose probs
from azureml.core.webservice import LocalWebservice

deployment_config = LocalWebservice.deploy_configuration(port=8890)
service = Model.deploy(ws, 'test-svc', [model], inference_config, deployment_config)

#test deployed service
print(service.run(input_data = json_data))

#troubleshoot issues by changing scoring file
#then reload wo having to redeploy
service.reload()
print(service.run(input_data = json_data))



##EXERCISE

#connect to ws
import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

from azureml.core import Experiment
from azureml.core import Model
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace = ws, name = "diabetes-training")
run = experiment.start_logging()
print("Starting experiment:", experiment.name)

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv('data/diabetes.csv')

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a decision tree model
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
model_file = 'diabetes_model.pkl'
joblib.dump(value=model, filename=model_file)
run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

# Complete the run
run.complete()

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Inline Training'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

print('Model trained and registered.')



#find what models are registered

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
    

#get model you want to deploy
model = ws.models['diabetes_model']
print(model.name, 'version', model.version)


#make folder for code and config files
import os

folder_name = 'diabetes_service'

# Create a folder for the web service files
experiment_folder = './' + folder_name
os.makedirs(experiment_folder, exist_ok=True)

print(folder_name, 'folder created.')

# Set path for scoring script
script_file = os.path.join(experiment_folder,"score_diabetes.py")


#scoring script
%%writefile $script_file
import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)


#define python enviro

from azureml.core.conda_dependencies import CondaDependencies 

# Add the dependencies for our model (AzureML defaults is already included)
myenv = CondaDependencies()
myenv.add_conda_package('scikit-learn')

# Save the environment config as a .yml file
env_file = os.path.join(experiment_folder,"diabetes_env.yml")
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

# Print the .yml file
with open(env_file,"r") as f:
    print(f.read())


#deploy, here to container instance
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = "diabetes-service"

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)


#check status
print(service.state)
print(service.get_logs())

# If you need to make a change and redeploy, you may need to delete unhealthy service using the following code:
#service.delete()

#get names of web services in ws
for webservice_name in ws.webservices:
    print(webservice_name)


#use web service

import json

x_new = [[2,180,74,24,21,23.9091702,1.488172308,22]]
print ('Patient: {}'.format(x_new[0]))

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Call the web service, passing the input data (the web service will also accept the data in binary format)
predictions = service.run(input_data = input_json)

# Get the predicted class - it'll be the first (and only) one.
predicted_classes = json.loads(predictions)
print(predicted_classes[0])

for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )


#get URL 
endpoint = service.scoring_uri
print(endpoint)    


#make HTTP request
import requests
import json

x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Set the content type
headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, input_json, headers = headers)
predicted_classes = json.loads(predictions.json())

for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )


 #delete service
 service.delete()
print ('Service deleted.')   
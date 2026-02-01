# House Price Predictor FastAPI App

This document will give description of the changes maded to the code from the course and how to build the container image for this fastapi app.

## Background

We have created a house price predictor package that takes historical data on house sales build a preprocessor and regression model. Both models are stored in the local host mlflow tracking system in the model registry as models moved to production after training.

This app will be the inference to those models. It will take in a request or batch request and make a prediction of the house price and confidence interval using our models. It will also provide shapley values for for the request or batch request indicting the contribution the feature had on the predicted price of the house for that sample.


## Pydantic base model

As our schema is built using pydantic BaseModel the dict function has been deprecated inplace of model_dump.  As we are using python have the mode set to **python** this means that the incoming data from this schema will be a python dictionary which we can convert to a pandas dataframe which we do in the inference file.


## Intructions to build Container Image for this FastAPI App 



Create the Dockerfile in api folder which is off the root of  (`house-prcice-predictor`). 

Following is all the information you would need to start building the container image for this app 


  * Base Image : `python:3.9`
  * To install dependencies: `pip install requirements.txt`
  * Port: `8000`
  * Launch Command : `uvicorn main:app --host 0.0.0.0 --port 8000`

Directory structure inside the container should look like this 

```
/app
  main.py
  schemas.py
  inference.py
  requirements.txt
  /src
    /house_price_predictor
     
```

when running podman ensure have the flag --network=host set so that the container can computicate wit the host machine to compunicate with mlflow and set the e envionment variable for mlflow tracking uri
- use idt when running as will do the followiong
 - i is a flag for interactive.  this allows you to send commands to the container via your terminal
 - d detatch. Runs the container in detached mode.this allows you to continue to use the terminal
 - t pseudo terminal.  provides a terminal interface within the container.

 ## main 
 This houses the code for the fast api.

 The **app** is an FastAPI object with the following parameters set at creation

 - title
 - description
 - version
 - contact: Left the contact of the people from the course but added mine as well as I've edited this
 - licence

 from fast api imported CORS middleware.  CORS(CROSS-ORIGIN RESCOURSE SHARING) refers to situations when a frontend running in a browser that communicates with a backend, and the backend is in a different "orgin" that the frontend

 **Origin**: is the combination of protocol (http, https), domain and port.  As result the following are all different origins

 - http://localhost
 - https://localhost
 - http://localhost:8000

 Even though that are all in the localhost because they are using different portocols or ports they are different "origins".

 
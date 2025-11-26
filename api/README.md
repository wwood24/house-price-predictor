# House Price Predictor FastAPI App

This document will give description of the changes maded to the code from the course and how to build the container image for this fastapi app.

## Background

We have created a house price predictor package that takes historical data on house sales build a preprocessor and regression model. Both models are stored in the local systems mflow tracking server with a backend to a postgress database. 

This app will be the inference will it will take in a request or batch request and make provide a prediction of the house price and confidence interval using our models.

## Pydantic base model

As our schema is built using pydantic BaseModel the dict function has been deprecated inplace of model_dump.  As we are using python have the mode set to **python** this means that the incoming data from this schema will be a python dictionary which we can convert to a pandas dataframe which we do in the inference file.


## Intructions to build Container Image for this FastAPI App 



Create the Dockerfile in the root of the source code (`house-prcice-predictor`). 

Following is all the information you would need to start building the container image for this app 


  * Base Image : `python:3.11-slim`
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
  /models
     /trained
         house_price_model.pkl
         preprocessor.pkl
```


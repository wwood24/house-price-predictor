# House Price Prediction system
Will outline the changes maded to this system. The system will be control by the house price predictor package and will have three services etl model which will be a container that trains preprocess and runs model expermients, the api container will be buit with fast api to make single and batch predictions loading the model from mlflow, streamlit app will be the users front exposer to the model which will send request to the fast api which will communicate with the mlflow model to make predictions and send back.

<details><summary><b> V1.0.0 (under concstruction) </b></summary>

Program will have three services

- etl_model
- fast api
- streamlit app

The fast api and streamlit app will be combined into a single docker compose which will be create with podman.


## etl model service

Here we break down how to build and run the container

to build the image

```bash
podman build --no-cache -t house_price_etl -f /etl_model_api/Dockerfile.etl
```


To run the container:
```bash
podman run --network=host  --rm -v /directory/data:/app/data -v /directory/logging:/app/logging -e DATA_DIR=/app/data -e LOG_DIR=/app/logging -e MLFLOW_TRACKING_URI=http://ipaddress:port -v /directory/mlflow/mlruns:/directory/mlflow/mlruns  house_price_etl:latest

```

-network=host: means that the container will share the network namespace and stack with the host machine we need this because we need container to be able to communicate with mlflow which is running on host machine not within the container.
- v is for volume this allows us to bind the source dir into the container dir so that we can communicate with data outside or send data outside the container.
-e enviornment variables we want to set at run time for the container to have.

## Fast API Container

Here we crated the following files:

- Dockerfile.api:
    - provides the instructions for building the container
- main.py:
    - builds the fast api app
- inference.py:
    - loads the preprosing and prediction model from mlflow
    - controls how to making single and batch inference
- requirments.txt
    - provides the additional packages we need to build the image

From the root directory this command will build our image:

```bash
podman build --no-cache -t house_price_api -f /api/Dockerfile.api .
```

-t: this is the tag we name this build
-f: is for file and points to the full path where to find the dockerfile to build the image. the "." after indicates the current working directory and that defines the build context when its processing the dockerfile in when its executing commands in the dockerfile like copy or add thus its important when creating the dockerfil to take that into account when writing the code as the copy and other commands will be affected by that to ensure correct files/directories are copied over.
- no-cache: instructs the command not to use any exisiting cached images for the build process.  Thus if we are rebuilding this image the process will start at the beginning.

To run the container.  For this run we have logging file and the mlflow outside the container as such need to map outside volumes into directories within the contain and pass env variables so that it can communicate with the outside mlflow service thus we use other flags when we run the services.

```bash
podman run -idt --network=host  -v SOURCE-LOG_DIR:CONTAINER-LOG-DIR -e LOG_DIR=CONTAINER-LOG-DIR -e MLFLOW_TRACKING_URI=http://mlflow:port -v /directory/mlruns:/directory/mlruns -p 8000:8000  house_price_api:latest
```

-network=host: means that the container will share the network namespace and stack with the host machine we need this because we need container to be able to communicate with mlflow which is running on host machine not within the container.
- v is for volume this allows us to bind the source dir into the container dir so that we can communicate with data outside or send data outside the container.
-e enviornment variables we want to set at run time for the container to have.

## Streamlit App

Created the app.py which contains all the required code to deploy the streamlit app and connect to the fast api app which will be running outside 

To build the image cd into the streamlit_app folder

```bash
podman build --no-cache -t house_price_app
```

in the docker file we have the command 

- CMD ["streamlit","run" ,"app.py","--server.address"=0.0.0.0]

we have it set to 0.0.0.0 and not 127.0.0.1 to allow things outside the container to reach the service running the container ie. the actual streamlit app needs to have it se to this to bind to the inside of the container the port we are exposing our system to.
</details>
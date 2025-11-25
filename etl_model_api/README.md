# ETL API Service
Here we will use our house_price_predictor package and wrap into a podman container.
The mlfow tracking service will run outside container localling and the container will communicate and log experiments into mlflow tracking server.

# main.py
This will be the file that the container will call as its CMD

# Dockerfile

- With this project the data and logs are housed within the local file system of this project and for clean small container as its already large with the packages we will mount volumes to the container at run time.

- As Mlflow is running locally will need to call it with environment variable of **http://host.containers.internal:5000**

- when one runs the command podman build and use . its key and important to know where your a running this command from because the **.** becomes the entry point for this app thus when one is copying folders into the container need to know with relation to where one is located in relation to the . and the folders needed.

## Key elements to go into a docker file
Here are some of hte key things to go into a docker file

- **FROM python:version-slim**:
 - this is the base image of the container and sets the foundation of your container provdiing pre-configured enviornment within a python install.
- **WORKDIR /folder**:
 - this sets the working directory within the image and thus any calls of RUN, COPY OR ADD will operatate realtive to this new working directory.
- **COPY**:
 - use this compand to copy something like a file or folder within the container
- **RUN pip install -r requirments.txt**: 
 - assuming one copied over a requirements.txt for additonal dependencies need this will install those packages within the container.
- **CMD []**:
 - this is the command that will run when calling podman/docker run

 ## commands to build and run container

 - **podman build -t tag_name -f /folder/Dockerfile .**:
  - this command will build the container image with tag_name as its tag one ues the -f if the docker file is not within the root of where one is running the command and "." will set this current directory your in as the relative path starting point which is key to remember when filling in the copy comands in the docker file.

- **podman run -v /data/:locactionincontainer -e some_env_variable tag_name:latest**:
 - this will run the container with a volume mounted and assigned ot the locationcontainer folder within container for reference within code tag_name:latest will run the latest container image with that tag_name


 Because I have special libraries need to add this to docker file:
 # Install OS dependencies for compiled Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*



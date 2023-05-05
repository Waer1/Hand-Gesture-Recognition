# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

RUN apt-get update

RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set the working directory to /app
WORKDIR /model

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /model/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# copy the target file to the container
COPY ./FinalCode /model/FinalCode

WORKDIR /model/

# Run app.py when the container launches
CMD [ "python" , "./FinalCode/main.py" ]


# docker run -it -v ./Dataset/:/model/Dataset -e "FEATURE_METHOD=0" -e "MODEL_METHOD=0" hand sh

# docker run -e "FEATURE_METHOD=0" -e "MODEL_METHOD=0" -v ./Dataset/:/model/Dataset hand > output

# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /train

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /train/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# copy the target file to the container
COPY ./run.py /train/files

# Run app.py when the container launches
CMD ["python", "app.py"]
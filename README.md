# Poetry dependencies:

The code in this project was tested with python 3.10 aswell as 3.11
The dependencies are managed with poetry. You can also install them manually with pip, but it is not recommended.
The exact versions of the dependencies are locked in the poetry.lock file.



# Non poetry dependencies:

Check https://github.com/KaiyangZhou/deep-person-reid and install the project into your projects environment.
You don't need to use conda like the original project, just install the requirements.txt file and run python setup.py install. 
Make sure it is added to your python path. 

# Models

Download the Re-Id models and put them in the respective model folders:
```
models/re-id/real50artificial50
models/re-id/real80
```


# Data

Download the image dataset and put the images in the data folder. The folder structure should be as follows:
```
data/boxes-raw
```


# Running the code

For replicating the results in the paper, run the python scripts named t**_experiments.py in the src folder.


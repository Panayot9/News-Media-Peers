# News-Media-Peers


### Setting it up
1. Clone the repo
```
>>> git clone https://github.com/Panayot9/News-Media-Peers.git
```

2. Set up virtual environment with Anaconda
```
>>> conda create --name mediapeers python=3.8
>>> conda activate mediapeers
>>> pip install -r requirements.txt
```

### Running MLflow
All the experiments will be logged in MLflow. To run the server and see the experiment results more clearly execute the following command:

```
>>> mlflow ui
```
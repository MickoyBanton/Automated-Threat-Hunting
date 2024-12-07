# Malicious Network Traffic Detection Using Reinforcement Learning

This repository implements a Deep Q-Learning model to identify malicious network traffic using the ALLFLOWMETER_HIKARI2021 dataset. The model is trained using the Stable-Baselines3 and leverages reinforcement learning techniques to adaptively detect anomalies in network traffic. The enviroment used was openAI's gymnasium.

# Dataset: ALLFLOWMETER_HIKARI2021

The ALLFLOWMETER_HIKARI2021 dataset contains labeled network traffic samples, making it an ideal resource for training and testing anomaly detection algorithms. It includes features like flow duration, packet sizes, and byte rates, along with corresponding labels indicating malicious or benign activity.  
  
To download this csv file go to 
```
https://zenodo.org/records/6463389
```

# Features

- Uses a pre-trained Deep Q-Network (DQN) model to classify network traffic as malicious or benign.
- Supports running the model directly without requiring a new environment setup.
- Modular code for easy modification and integration with other datasets or architectures.

# Requirements
Ensure the following libraries are installed before running the project:

- Python 3.8+
- stable-baselines3
- gymnasium
- numpy
- scikit-learn

Install the dependencies using:
```
pip install -r requirements.txt
```

# Files in the Repository
- traffic_dqn_model.zip: Pre-trained DQN model for detecting malicious traffic.
- Automated_Threat_Hunting.ipynb: Script for training the DQN model.
- test_model.ipynb: Script showing an example of how to test the model model on new data.
- README.md: Project documentation (this file).

# How to Use the Pre-Trained Model
To use the pre-trained model without creating an environment:

1. Clone this repository:
```
git clone https://github.com/MickoyBanton/Automated-Threat-Hunting.git <br>
```
```
cd Automated-Threat-Hunting
```

2. Load the model and run it on test data:

```
from stable_baselines3 import DQN
import numpy as np

# Load the pre-trained model
model = DQN.load("traffic_dqn_model")

# Test data (example observation; replace with real data)
test_data = np.array([0.1, 0.5, ..., 0.3])  # A 20-dimensional vector

# Get predicted action (0: benign, 1: malicious)
action, _ = model.predict(test_data)
print(f"Predicted action: {action}")

```
 - This approach avoids creating a custom environment since you're working directly with observations.  

3. Replace test_data with the actual input features you want to classify.

# Model Explanation
The model observes a 20-dimensional feature space and takes actions (benign or malicious) based on its policy. It was trained on simulated network traffic with reward signals incentivizing correct classifications.

# Contributions
Feel free to open issues or submit pull requests to improve this project!

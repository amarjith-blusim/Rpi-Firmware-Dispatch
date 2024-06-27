# Firmware of Mat_V2
### Only update the code in process.py for preprocessing and prediciton.
## Main Code
Run the `main_v2_mat.py` script for data collection and call functions for HR and MQTT. The preprocessing and prediction occur in the `process.py` file.

## Process Code
The `process.py` code contains preprocessing and model prediction for Presence, RR, BP, and SP.

## Function Code
The `function_v2.py` code contains functions necessary for the prediction of HR and processing/uploading data to Amazon AWS DynamoDB.

## Configurations
The `config_v2.py` file holds configurations, including MQTT certificates, sample rate, and other parameters essential for HR prediction and preprocessing. It also includes IMF bands for BP and RR, frequency cuts, and threshold values for HR prediction.

## Simulation Data
Utilize the `simulation.py` script to simulate data from a recorded CSV file, mimicking the functions of actual recording.

## 4G Connectivity
The `net.py` file contains code that connects with `ttyUSB5`, handling 4G connectivity, network setup using `nmcli`, and connection methods.

## Models
Ensure there is a folder named `models` in the same directory as the code.

## Filepaths
RPi-firmware
|-- /BLS_certs
|-- /models


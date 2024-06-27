import time 
import os
# ADC Configurations 
SAMPLE_RATE = 490

# Data configuration
DURATION_UNIT_DATA = 60 #seconds - 1 minute data is processed everytime

# IMF Settings for BP, RR, SC
IMF_BP = 1
IMF_RR = 2 

#Frequency cut for BP and RR
highcut = 6
lowcut = 0.01

# MQQT
basedir = f"{os.path.dirname(os.path.abspath(__file__))}/BLS_certs"
sid = "BLS01"
endpoint='a3pb4fdke2uq4b-ats.iot.ap-southeast-1.amazonaws.com' ## required endpoint
port=8883 
cert=basedir + '/' + sid + '/BLS01-certificate.pem.crt' ## path to .crt file
key=basedir + '/' + sid + '/BLS01-private.pem.key' ## path to key file
root_ca=basedir + '/' + sid + '/AmazonRootCA1.pem' ## path to .pem file
client_id= sid + "_main" # + str(uuid4()) ## a defined client_id
topic='bls/' + sid + '/main' ## name of the topic
use_websocket=False
aws_region='ap-southeast-1' ## server region 
proxy_host = None
proxy_port= 8080

#Parameters for S3


# HR 
V_CHANNEL_THRESHOLD = 5
Q_LIMIT=20
HR_WINDOW_SIZE = 249
HR_WINDOW_SHIFT = 2
HR_WINDOWS_TO_SKIP = 5
HR_MOVING_AVERAGE_PARAM = 100

HR_MED_WINDOW_SIZE = 40 # ~5 secs
HR_MED_WINDOW_SHIFT = 20 # ~3 secs
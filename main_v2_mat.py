# Read 1 min data from ADC and run functions to sent HR, RR, SC, SP, BP and PR using MQQT
# The code should be able to connect to 4G (if not connected), should be able to handle errors, and record data only using battery - operated through relay 
# All the configurations are present in the config file

from config_v2 import *
from functions_v2 import * 
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn
#Switiching to DC intialisation
import RPi.GPIO as GPIO
relay_dc=18
if GPIO.getmode() is None:
    GPIO.setmode(GPIO.BOARD)
GPIO.setup(relay_dc, GPIO.OUT)
# ADC Intialisation
i2c = busio.I2C(board.SCL, board.SDA)
_ads = [ADS.ADS1015(i2c,address=0x48), ADS.ADS1015(i2c,address=0x49), ADS.ADS1015(i2c,address=0x4b), ADS.ADS1015(i2c,address=0x4a)]
chan = [[],[],[],[]]
for i in range(4):
    _ads[i].gain = 16
    _ads[i].mode = Mode.CONTINUOUS
    _ads[i].data_rate = SAMPLE_RATE
    chan[i] = AnalogIn(_ads[i], ADS.P0)

data_buff_list = [[],[],[],[]] # will store 2D array [channel number][signal data] - to define it globally for easy access of all functions


if __name__ == "__main__":
    # Initiallising mqtt
    mqtt_connect=mqtt_init()
    end_time = time.time() + DURATION_UNIT_DATA
    data_buff_list = [[],[],[],[]]
    # Collecting data using DC
    GPIO.output(relay_dc, True)
    while (end_time > time.time()):
        # Reads ADC Data
        for i in range(4):
            data_buff_list[i].append(chan[i].value)
    GPIO.output(relay_dc, False)
    print(np.array(data_buff_list).shape)

    # Heart Rate Estimation
    hr_estimate_voting(data_buff_list,mqtt_connect) # ~ 3 sec
    
    # Respiration Rate, Bloop Pressure and Sleep Cycle Prediction
    expected_samples = SAMPLE_RATE * DURATION_UNIT_DATA
    interpolated = np.zeros((4, expected_samples))
    filtered = pd.DataFrame()
    imfs = [pd.DataFrame()]*4
    ins_phase_bp = [pd.DataFrame()]*4
    ins_phase_rr = [pd.DataFrame()]*4
    for i in range(4):
        interpolated[i] = interpolate(data_buff_list[i],expected_samples)
        filtered['buttered_channel'+str(i+1)] = [butter_bandpass_filter(interpolated[i], lowcut, highcut, SAMPLE_RATE, order=3)]
        imfs[i] = imf(filtered['buttered_channel'+str(i+1)],SAMPLE_RATE)
        ins_phase_bp[i] = inp(imfs[i][IMF_BP])
        ins_phase_rr[i] = inp(imfs[i][IMF_RR])

    #BP model loading and prediction
    # predict_bp(ins_phase_bp,mqtt_connect)

    # # RR model loading and prediction
    # predict_rr(ins_phase_rr,mqtt_connect)

    # Presence Detection
    # presence = predict_pr(data_buff_list)
    # print(presence)
    # if presence:
    #     print(np.array(data_buff_list).shape)
    #     #heart_beat(mqtt_connect)

    #     # Heart Rate Estimation
    #     hr_estimate_voting(data_buff_list,mqtt_connect) # ~ 3 sec
        
    #     # Respiration Rate, Bloop Pressure and Sleep Cycle Prediction
    #     expected_samples = SAMPLE_RATE * DURATION_UNIT_DATA
    #     interpolated = np.zeros((4, expected_samples))
    #     filtered = pd.DataFrame()
    #     imfs = [pd.DataFrame()]*4
    #     ins_phase_bp = [pd.DataFrame()]*4
    #     ins_phase_rr = [pd.DataFrame()]*4
    #     for i in range(4):
    #         interpolated[i] = interpolate(data_buff_list[i],expected_samples)
    #         filtered['buttered_channel'+str(i+1)] = [butter_bandpass_filter(interpolated[i], lowcut, highcut, SAMPLE_RATE, order=3)]
    #         imfs[i] = imf(filtered['buttered_channel'+str(i+1)],SAMPLE_RATE)
    #         ins_phase_bp[i] = inp(imfs[i][IMF_BP])
    #         ins_phase_rr[i] = inp(imfs[i][IMF_RR])

    #     #BP model loading and prediction
    #     predict_bp(ins_phase_bp,mqtt_connect)

    #     # RR model loading and prediction
    #     predict_rr(ins_phase_rr,mqtt_connect)
    # else:
    #     print("Empty Bed")

    

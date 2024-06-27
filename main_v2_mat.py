# Read 1 min data from ADC and run functions to sent HR, RR, SC, SP, BP and PR using MQQT
# The code should be able to connect to 4G (if not connected), should be able to handle errors, and record data only using battery - operated through relay 
# All the configurations are present in the config file
##########################################################
# Update process.py to tune and adjust the preprocessing #
# of all the different ML models -  SP,BP,RR,PR          #
##########################################################
from config_v2 import *
from functions_v2 import * 
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn
# Model prediction
from process import *
# Switiching to DC intialisation
import RPi.GPIO as GPIO
#Network Connectivity
from net import *
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
    i = 0
    while True:
        # Checking network connectivity
        net_main()
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
        # For simulation using data from csv file
        # from simulation import *
        # data_buff_list=data() 
        # Presence Detection
        presence = predict_pr(data_buff_list)
        print(presence)
        if presence:
            print(np.array(data_buff_list).shape)
            #heart_beat(mqtt_connect)
            # Heart Rate Estimation
            hr_estimate_voting(data_buff_list,mqtt_connect) # ~ 3 sec
            
            # Respiration Rate, Bloop Pressure and Sleep Cycle Prediction
            expected_samples = SAMPLE_RATE * DURATION_UNIT_DATA
            interpolated = np.zeros((4, expected_samples))
            filtered = pd.DataFrame()
            imfs = [pd.DataFrame()]*4
            ins_phase_bp = [pd.DataFrame()]*4
            ins_phase_rr = [pd.DataFrame()]*4
            for k in range(4):
                interpolated[k] = interpolate(data_buff_list[k],expected_samples)
                filtered['buttered_channel'+str(k+1)] = [butter_bandpass_filter(interpolated[k], lowcut, highcut, SAMPLE_RATE, order=3)]
                imfs[k] = imf(filtered['buttered_channel'+str(k+1)],SAMPLE_RATE)
                ins_phase_bp[k] = inp(imfs[k][IMF_BP])
            ins_phase_bp_array = [[float(val) for val in sublist[0]] for sublist in ins_phase_bp]

            #BP model loading and prediction
            if i == 1:
                print('BP')
                # Concatenate row-wise
                data_buff_list_bp = np.concatenate((data_buff_list_bp, ins_phase_bp_array), axis=1)
                print(data_buff_list_bp.shape)
                bp_sys, bp_dia = predict_bp(data_buff_list_bp)
                message = { "sid": sid, "batch_dur": 60, "ty": "SYS", "val": bp_sys, "val_len": len(bp_sys) }
                publish_to_server(topic,message, mqtt_connect)
                message = { "sid": sid, "batch_dur": 60, "ty": "DIA", "val": bp_dia, "val_len": len(bp_dia) }
                publish_to_server(topic,message, mqtt_connect)
                i = 0
            else:
                print('BP+')
                shape = ins_phase_bp_array.shape
                data_buff_list_bp = np.zeros(shape)  # Initialize with zeros
                # Now copy the new data
                data_buff_list_bp = np.copy(ins_phase_bp_array)
                i += 1
                

            # RR model loading and prediction
            pred_rr=predict_rr(interpolated)
            message = { "sid": sid, "batch_dur": 60, "ty": "RR", "val": pred_rr, "val_len": len(pred_rr) }
            publish_to_server(topic,message, mqtt_connect)

            # SP model loading and prediction
            pose = sleep_pose(interpolated)
            message = { "sid": sid, "batch_dur": 60, "ty": "SP", "val": pose, "val_len": len(pose) }
            publish_to_server(topic,message, mqtt_connect)

            # SC
            sc_data = filtered.to_numpy()
            # Extract individual arrays
            sc_data = sc_data[0]
            # Stack the arrays vertically to form a 2D array
            sc_data = (np.vstack(sc_data)).T
            sc = sleep_cycle(sc_data)
            message = { "sid": sid, "batch_dur": 60, "ty": "SP", "val": sc, "val_len": len(sc) }
            publish_to_server(topic,message, mqtt_connect)
        else:
            print("Empty Bed")
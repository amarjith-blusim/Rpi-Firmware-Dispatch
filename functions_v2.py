import numpy as np
from awscrt import io, mqtt
from awsiot import mqtt_connection_builder
import sys
import json
import time
from config_v2 import *
import logging
import traceback
from select import select
from scipy import signal as sgn
import pandas as pd
from scipy.interpolate import PchipInterpolator
from PyEMD import EMD
from PyEMD import EEMD
from scipy.signal import butter, lfilter, hilbert
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dense, BatchNormalization
from datetime import datetime, timedelta
import tensorflow as tf
from keras.models import load_model
import joblib
##################################################
# Functions for  HB, HR, RR, SC, BP, SP, PR
##################################################
def heart_beat(mqtt_connect):
    # Send heart_beat to MQTT
    message = { "sid": sid, "batch_dur": 60, "ty": "HB", "val": [1], "val_len": 1 } #hardcoded value 1
    publish_to_server(topic,message, mqtt_connect)

def hr_estimate_voting(signals, mqtt_connect):
    try:
        start_time = time.time()
        hr_lists = [[], [], [], []]
        for i in range(4):
            print("2")
            single_channel(signals[i], i, hr_lists)
            

        
        V_WIN = 1
        
        n_bpm = len(hr_lists[0])
        v_current_window = 0

        hr_compliance = []

        while(v_current_window + V_WIN < n_bpm):
            v_win_data = [[], [], [], []]
            for channel in range(4):
                for j in range(V_WIN):
                    v_win_data[channel].append(hr_lists[channel][j + v_current_window])                 
            
            v_agree = 0
            v_agree_index = [0]

            # Compare with the rest of the channel
            # Step 1: Referrence channel is 0. Compare with 1 2 and 3. if any of the values are within the threshold of the referrence channel, store their index.
            for i in range(3):
                if (v_win_data[0][0] - V_CHANNEL_THRESHOLD <= v_win_data[i+1][0] and v_win_data[i+1][0] <= v_win_data[0][0] + V_CHANNEL_THRESHOLD):
                    v_agree += 1
                    v_agree_index.append(i+1)

            # Step 2: If there are 3 channels that agree on each other, it say it complies.
            comply = False
            if (v_agree >= 2):
                comply = True
            
            # Step 2a: If it does not comply, we change referrence channel to channel 1 and repeat the process
            if (comply == False):
                v_agree = 0
                for i in range(2):
                    if (v_win_data[1][0] - V_CHANNEL_THRESHOLD <= v_win_data[i+2][0] and v_win_data[i+2][0] <= v_win_data[1][0] + V_CHANNEL_THRESHOLD):
                        v_agree += 1
                if (v_agree == 2):
                    comply = True
                    v_agree_index = [1, 2, 3]
            
            # Step 3: if it complies, we find the rounded average heartbeat. 
            # Reason for average
            #   - Average simplifies the heuristics we use. If all 3 or 4 channels are the same value, its average is the same value
            #   - one differ (ie. 71 71 71 70 or 71 71 70 XX) then its average will take the mode (most repeated number)
            #   - 2 or more differ (ie. 71 71 70 70) then its average will give the fairest result as we have no idea which is the correct one.
            if (comply == True):
                sum = 0
                for i in v_agree_index:
                    sum += v_win_data[i][0]

                hr_bpm = round(sum / len(v_agree_index), 2)
                hr_compliance.append(hr_bpm)
            else:
                hr_compliance.append(0)

            v_current_window += V_WIN
            
        end_time = time.time()
        duration = end_time - start_time
        print(hr_compliance)
        print("Total time elapsed = " + str(round(duration, 2)) + " sec")
        print("=============\n")
        message = { "sid": sid, "batch_dur": 60, "ty": "HR", "val": hr_compliance, "val_len": len(hr_compliance) }
        publish_to_server(topic,message, mqtt_connect)
    except:
        #disconnect(mqtt_connect)
        logging.error("Error in Heart rate function .")
        # traceback.print_exc()


##################################################
# General functions for EMD, Bandpass, Inst. Phase
##################################################

def interpolate(arr,expected_sps):
    pchip_interpolator = PchipInterpolator(np.arange(len(arr)), arr)
    x_interp = np.linspace(0, len(arr) - 1, expected_sps)
    y_interp = pchip_interpolator(x_interp)
    return y_interp.tolist()

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def imf(arr,fs):
  _out = [[],[],[],[]]
  for i in arr:
    for j in range(4):
      _out[j].append(EMD().emd(i, (np.arange(0, len(i)) / fs))[j])
  return _out

def inp(arr):
  _out = []
  for i in arr:
    _out.append(np.angle(hilbert(i)))
  return _out


##################################################
# General functions for calculating the Heart rate
##################################################

# Single channel Heart Rate calculation
def single_channel(signal, signal_no, result):
    signal_len = len(signal) # number of data points inside the array
    #print(signal_len)
    cepstrum_from_windows = []
    start = 0
    while (start+HR_WINDOW_SIZE) <= signal_len: # taking a 249 signal data chunks
        quefrency_vector, cepstrum = get_cepstrum(signal[start:start+HR_WINDOW_SIZE], SAMPLE_RATE)
        cepstrum_from_windows += [np.abs(cepstrum)]
        start += int(HR_WINDOW_SHIFT)
    cepstrum_from_windows = [i[find_closest_index(quefrency_vector, 0.7):] for i in cepstrum_from_windows]
    #print(len(cepstrum_from_windows),"3")
    quefrency_vector = quefrency_vector[find_closest_index(quefrency_vector, 0.7):]
    max_of_windows = []
    num_of_windows = len(cepstrum_from_windows)
    #print(len(quefrency_vector),len(cepstrum),len(cepstrum_from_windows))
    #print(num_of_windows,"4")
    curr_window = 0
    while curr_window < num_of_windows:
        # find the peaks, and store their values and indices
        peaks_index_in_window, _ = sgn.find_peaks(cepstrum_from_windows[curr_window])
        if (len(peaks_index_in_window) == 0):
            curr_window += HR_WINDOWS_TO_SKIP
            continue

        peaks_value_and_indices = []
        for peak_index in peaks_index_in_window:
            peaks_value_and_indices += [ (quefrency_vector[peak_index], cepstrum_from_windows[curr_window][peak_index]) ]

        peaks_value_and_indices.sort(key=lambda tup: tup[1], reverse=True)
        selected_peaks_in_window = [i[0] for i in peaks_value_and_indices[:1]]
        selected_peaks_in_window.sort(reverse=True)
        selected_peaks_in_window = [selected_peaks_in_window[0]]

        max_of_windows += [selected_peaks_in_window]
        curr_window += HR_WINDOWS_TO_SKIP

    #print(len(max_of_windows))
    mov_average = moving_average([maxes[-1] for maxes in max_of_windows], HR_MOVING_AVERAGE_PARAM)
    #print(mov_average,len(mov_average))

    hr_np = ((1/mov_average)*60)
    # Reduce data using median
    hr_list = []

    start_med = 0
    while (start_med+ HR_MED_WINDOW_SIZE) <= len(hr_np):
        med_window = hr_np[start_med:start_med+HR_MED_WINDOW_SIZE]
        med_val = round(np.median(med_window))
        hr_list.append(med_val)
        # print(med_val)
        start_med += HR_MED_WINDOW_SHIFT

    # print("Length: " + str(len(hr_list)))
    result[signal_no] = hr_list

# Calculates the sample number at a given minute based on sample rate.
def minute(minute):
    return int(minute * 60 * SAMPLE_RATE)

# Calculates samples of Frequency distances
def get_frequency_distance_in_samples(f, distance):
    return distance // (f[1] - f[0])

# Calculates Fourier Transform
def get_fourier_transform(signal, sample_freq, ham=1):
    frame_size = len(signal)
    windowed_signal = np.hamming(frame_size) * signal
    if ham==0:
        windowed_signal = signal
    dt = 1/sample_freq
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    X = np.fft.rfft(windowed_signal)
    return (freq_vector, X)

# Calculates Cepstrum for a batch of data
def get_cepstrum(signal, sample_freq):
    freq_vector, X = get_fourier_transform(signal, sample_freq)
    log_X = np.log(np.abs(X))
    cepstrum = np.fft.rfft(log_X)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_X.size, df)
    return (quefrency_vector, cepstrum)

# Find the index of the closest Frequency bin in the list of quefrencies returned by get_cepstrum
def find_closest_index(search_list, target):
    for i in range(len(search_list)):
        if target < search_list[i]:
            return i - 1 if i != 0 else 0

# Calculaters the moving average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

################################
# Functions for MQTT protocol
################################

# Callback when connection is accidentally lost.
def on_connection_interrupted(connection, error, **kwargs):
    print("Connection interrupted. error: {}".format(error))

# Callback when an interrupted connection is re-established.
def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print("Connection resumed. return_code: {} session_present: {}".format(return_code, session_present))

    if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
        print("Session did not persist. Resubscribing to existing topics...")
        resubscribe_future, _ = connection.resubscribe_existing_topics()

        # Cannot synchronously wait for resubscribe result because we're on the connection's event-loop thread,
        # evaluate result with a callback instead.
        resubscribe_future.add_done_callback(on_resubscribe_complete)

# Callback for resubscribing 
def on_resubscribe_complete(resubscribe_future):
        resubscribe_results = resubscribe_future.result()
        print("Resubscribe results: {}".format(resubscribe_results))

        for topic, qos in resubscribe_results['topics']:
            if qos is None:
                sys.exit("Server rejected resubscribe to topic: {}".format(topic))

# Callback when the subscribed topic receives a message
def on_message_received(topic, payload, dup, qos, retain, **kwargs):
    print("Received message from topic '{}': {}".format(topic, payload))

# Initialisation of the MQTT connection
def mqtt_init():
    # Spin up resources
    event_loop_group = io.EventLoopGroup(1)
    host_resolver = io.DefaultHostResolver(event_loop_group)
    client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

    proxy_options = None
    
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=endpoint,
        port=port,
        cert_filepath=cert,
        pri_key_filepath=key,
        client_bootstrap=client_bootstrap,
        ca_filepath=root_ca,
        on_connection_interrupted=on_connection_interrupted,
        on_connection_resumed=on_connection_resumed,
        client_id=client_id,
        clean_session=False,
        keep_alive_secs=30,
        http_proxy_options=proxy_options)

    print("Connecting to {} with client ID '{}'...".format(endpoint, client_id))

    connect_future = mqtt_connection.connect()

    # Future.result() waits until a result is available
    connect_future.result()
    print("Connected!")

    # Subscribe
    print("Subscribing to topic '{}'...".format(topic))
    subscribe_future, packet_id = mqtt_connection.subscribe(
        topic=topic,
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=on_message_received)

    subscribe_result = subscribe_future.result()
    print("Subscribed with {}".format(str(subscribe_result['qos'])))
    return mqtt_connection 

# function to publish into the server endpoint 
def publish_to_server(topic,message,mqtt_connection):
    
    message_json = json.dumps(message)
    mqtt_connection.publish(
        topic=topic,
        payload=message_json,
        qos=mqtt.QoS.AT_LEAST_ONCE)
    print("Publishing message to topic '{}': {}".format(topic, message))

# function to disconnect fromn the server
def disconnect(mqtt_connection):
    print("Disconnecting...")
    disconnect_future = mqtt_connection.disconnect()
    disconnect_future.result()
    print("Disconnected!")


####################################
# Functions for Presence Detection #
####################################

def predict_pr(data_buff_list):
    model_filename = '/home/gokulmc/Desktop/models/presence/presence.joblib'
    model_pr = joblib.load(model_filename)
    chan1 = np.std(data_buff_list[0][:490])
    chan2 = np.std(data_buff_list[1][:490])
    chan3 = np.std(data_buff_list[2][:490])
    chan4 = np.std(data_buff_list[3][:490])
    
    # Assuming chan1, chan2, chan3, chan4 are your features
    features = np.array([[chan1, chan2, chan3, chan4]])

    # Make predictions
    prediction_pr = model_pr.predict(features)
    #print(prediction_pr)
    return prediction_pr[0] # 1 for presence and 0 for no presence
###############################
# Functions for BP prediction #
###############################
def predict_bp(ins_phase_bp,mqtt_connect):
    merged_ipc_bp = np.stack((ins_phase_bp[0:4]), axis=1)
    merged_ipc_bp = np.swapaxes(merged_ipc_bp, 1, 2)
    model_bp = load_model('/home/gokulmc/Desktop/models/blood_pressure/BP.h5') # need to add model filename
    predictions_bp = model_bp.predict(merged_ipc_bp)
    #combined = pd.DataFrame(predictions_bp, columns=['Systolic(predicted)', 'Diastolic(predicted)']) # need to add model filename
    bp_sys = ["{:.2f}".format(predictions_bp[0][0])]
    bp_dia = ["{:.2f}".format(predictions_bp[0][1])]
    print(bp_sys,bp_dia)
    message = { "sid": sid, "batch_dur": 60, "ty": "SYS", "val": bp_sys, "val_len": len(bp_sys) }
    publish_to_server(topic,message, mqtt_connect)
    message = { "sid": sid, "batch_dur": 60, "ty": "DIA", "val": bp_dia, "val_len": len(bp_dia) }
    publish_to_server(topic,message, mqtt_connect)
###############################
# Functions for RR prediction #
###############################
def predict_rr(ins_phase_rr,mqtt_connect):
    model_rr = load_model('/home/gokulmc/Desktop/models/respiration_rate/RR.h5')
    # Function to divide each array into six equal parts
    def divide_into_six_parts(array):
        split_arrays = np.array_split(array[0], 6)  # Use array[0] to handle the additional list
        return [np.array([part]) for part in split_arrays]

    # Apply the function to each array in the list
    divided_arrays = [divide_into_six_parts(inner_array) for inner_array in ins_phase_rr]
    # Transpose the result to have six lists
    divided_arrays = list(map(list, zip(*divided_arrays)))
    # Print the divided arrays
    predictions=[]
    for idx, divided_array in enumerate(divided_arrays):
        #print(f"List {idx + 1}: {divided_array}")
        merged_ipc = np.stack((divided_array[0:4]), axis=1)
        merged_ipc = np.swapaxes(merged_ipc, 1, 2)
        prediction=model_rr.predict(merged_ipc)
        predictions.extend(float(value) for value in prediction.flatten())
    print(predictions)
    message = { "sid": sid, "batch_dur": 60, "ty": "RR", "val": predictions, "val_len": len(predictions) }
    publish_to_server(topic,message, mqtt_connect)
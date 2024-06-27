import numpy as np
import joblib
import os
import tensorflow as tf
import pandas as pd
import pywt
from scipy.signal import resample, hilbert
from scipy.stats import mode
from config_v2 import SAMPLE_RATE
from sklearn.preprocessing import LabelEncoder
####################################
# Functions for Presence Detection #
####################################
def predict_pr(data_buff_list):
    model_filename = f"{os.path.dirname(os.path.abspath(__file__))}/models/presence.joblib"
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
def predict_bp(ins_phase_bp):
    merged_ipc_bp = np.stack((ins_phase_bp[0:4]), axis=1)
    merged_ipc_bp= np.swapaxes(merged_ipc_bp, 0, 1)
    merged_ipc_bp = np.reshape(merged_ipc_bp.T, (-1, 58800, 4))
    model_bp = tf.keras.models.load_model(f"{os.path.dirname(os.path.abspath(__file__))}/models/BP.h5") # need to add model filename
    predictions_bp = model_bp.predict(merged_ipc_bp)
    #combined = pd.DataFrame(predictions_bp, columns=['Systolic(predicted)', 'Diastolic(predicted)']) # need to add model filename
    bp_sys = ["{:.2f}".format(predictions_bp[0][0])]
    bp_dia = ["{:.2f}".format(predictions_bp[0][1])]
    print(bp_sys,bp_dia)
    return bp_sys,bp_dia

###############################
# Functions for RR prediction #
###############################
def predict_rr(data_buff_list):
    arg_amplitude_envelopes = []
    differentiated_arg_env_signals = []
    fs = 490
    for channel in range(4):
        analytic_signal = hilbert(data_buff_list[channel]) # Apply Hilbert transform
        amplitude_envelope = np.abs(analytic_signal) # Extract the amplitude envelope
        arg_amplitude_envelope = np.angle(analytic_signal) # Calculate the arctangent of the amplitude envelope
        phase_signal = np.angle(analytic_signal) #Find phase 
        differentiated_arg_env_signal = np.gradient(phase_signal) / (2 * np.pi) # Differentiate the arctangent of the amplitude envelope signal and divide it by 2*pi
        
        arg_amplitude_envelopes.append(arg_amplitude_envelope)
        differentiated_arg_env_signals.append(differentiated_arg_env_signal)
    # Convert lists to numpy arrays
    arg_amplitude_envelopes = np.array(arg_amplitude_envelopes)
    differentiated_arg_env_signals = np.array(differentiated_arg_env_signals)

    # Initialize lists to store results per channel
    max_frequency_per_minute_all = []
    rr_values_per_minute_all = []

    # Loop through each channel
    for channel_index in range(len(differentiated_arg_env_signals)):
        differentiated_arg_env_signal2 = differentiated_arg_env_signals[channel_index]

        # Initialize a list to store the maximum frequency per minute
        max_frequency_per_minute = []
        rr_values_per_minute = []

        # Calculate the frequency for each second and find the maximum per minute
        for i in range(0, len(differentiated_arg_env_signal2), fs):
            segment = differentiated_arg_env_signal2[i:i+fs]
            zero_crossings = len(np.where(np.diff(np.sign(segment)))[0])
            frequency = zero_crossings / 2  # Frequency per second

            # Append the frequency to the list if its the start of a new minute
            if i % (fs * 60) == 0:
                max_frequency_per_minute.append(frequency)
            else:
                # Update the maximum frequency if the current frequency is higher
                max_frequency_per_minute[-1] = max(max_frequency_per_minute[-1], frequency)

        # Calculate the respiration rate for each minute from the maximum frequency
        for freq in max_frequency_per_minute:
            rr = (freq*60)/200  # Calculate respiration rate
            rr_values_per_minute.append(rr)  # Store RR value

        # Append results for the current channel to the overall list
        max_frequency_per_minute_all.append(max_frequency_per_minute)
        rr_values_per_minute_all.append(rr_values_per_minute)

    # Calculate the mode of RR values for each minute across all channels
    num_minutes = len(rr_values_per_minute_all[0])  # Number of minutes to consider (same for all channels)
    mode_rr_per_minute = []

    for minute in range(num_minutes):
        rr_values_minute = [rr_values[minute] for rr_values in rr_values_per_minute_all]
        mode_rr, _ = mode(rr_values_minute)
        mode_rr_per_minute.append(mode_rr)

    return mode_rr
###############################
# Functions for SP prediction #
###############################
def sleep_pose(interpolated):
    def compute_features(bcg_signal, wavelet='db4', level=4):
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(bcg_signal, wavelet, level=level)
        
        # Reconstruct the detailed coefficients at levels 3 and 4
        detail_3 = pywt.waverec([coeff if i == len(coeffs) - 3 else np.zeros_like(coeff) for i, coeff in enumerate(coeffs)], wavelet)
        detail_4 = pywt.waverec([coeff if i == len(coeffs) - 4 else np.zeros_like(coeff) for i, coeff in enumerate(coeffs)], wavelet)
        
        # Ensure the reconstructed signals are the same length as the original signal
        min_length = min(len(detail_3), len(detail_4), len(bcg_signal))
        detail_3 = detail_3[:min_length]
        detail_4 = detail_4[:min_length]
        bcg_signal = bcg_signal[:min_length]
        
        # Combine detailed coefficients
        reconstructed_signal = detail_3 + detail_4
        
        # Adaptive thresholding (example using simple threshold)
        threshold = np.std(reconstructed_signal) * 0.5
        peaks = np.where(reconstructed_signal > threshold)[0]
        
        # Compute features x0, x1, x2, x3
        x0 = np.mean(reconstructed_signal[peaks])  # Mean of peak values
        x1 = np.std(reconstructed_signal[peaks])   # Standard deviation of peak values
        x2 = np.max(reconstructed_signal[peaks])   # Maximum peak value
        x3 = np.min(reconstructed_signal[peaks])   # Minimum peak value
        
        return np.array([x0, x1, x2, x3])
    # Example usage
    sampling_rate = 490  # samples per second
    segment_duration = 5  # seconds
    segment_length = sampling_rate * segment_duration

    # Assuming interpolated is your BCG data array
    bcg_df = interpolated.T  # Adjust with your actual data source

    # Store features
    data = []

    num_segments = len(bcg_df) // segment_length

    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        
        segment_data = {}  # Initialize empty dictionary for segment data
        
        for channel in range(bcg_df.shape[1]):  # Iterate through each channel (assuming each column represents a channel)
            segment = bcg_df[:, channel][start_idx:end_idx]  # Extract data for the current channel
            
            features = compute_features(segment)
            
            # Store features for the current channel
            for j, feature in enumerate(features):
                segment_data[f'Channel{channel + 1}_x{j}'] = feature
        
        data.append(segment_data)

    # Create a DataFrame from the collected data
    features_df = pd.DataFrame(data)

    # Create the initial DataFrame (if needed)
    df = features_df

    # Initialize the new DataFrame dictionary (if needed)
    new_data = {
        f'Channel{i + 1}': [] for i in range(bcg_df.shape[1])
    }

    # Loop through each row in the original DataFrame
    for index, row in df.iterrows():
        # For each channel, create an array of the features and append
        for ch in range(bcg_df.shape[1]):
            channel_features = [row[f'Channel{ch + 1}_x{j}'] for j in range(4)]  # Assuming x0 to x3 are the feature names
            new_data[f'Channel{ch + 1}'].append(channel_features)

    # Create the new DataFrame (if needed)
    new_df = pd.DataFrame(new_data)

    # Extract features and labels
    features = ['Channel1', 'Channel2', 'Channel3', 'Channel4']
    X = new_df[features].values
    #y = main_df['Pose'].values

    # Convert lists to arrays
    X = np.array([np.array([np.array(channel) for channel in row]) for row in X])

    # Normalize the data
    X = X / np.max(X)

    X = np.reshape(X, (X.shape[0], X.shape[2], X.shape[1]))
    model = tf.keras.models.load_model(f"{os.path.dirname(os.path.abspath(__file__))}/models/pose.h5")

    # Make predictions
    predictions = model.predict(X)
    prediction = np.argmax(predictions, axis=1)

    
    # Example labels for the LabelEncoder (this should be the same as used during training)
    labels = ['left','right' ,'supine','prone']

    # Step 4: Initialize and fit the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Step 5: Use argmax to find the index of the highest predicted probability
    predicted_class_index = np.argmax(predictions, axis=-1)

    # Step 6: Map the index back to the original label
    predicted_label = label_encoder.inverse_transform(predicted_class_index)

    from collections import Counter

    # Input array
    array = np.array(predicted_label)

    # Define window size
    window_size = 12

    # Function to get the most frequent value in a window
    def most_frequent(arr):
        count = Counter(arr)
        return count.most_common(1)[0][0]

    # Process array in windows
    result = []

    for i in range(0, len(array), window_size):
        window = array[i:i + window_size]
        if len(window) == window_size:
            result.append(most_frequent(window))
    
    return result

###############################
# Functions for SC prediction #
###############################

def sleep_cycle(data):
    sc_labels = ['Core','Deep','REM']
    sc_model = tf.keras.models.load_model(f"{os.path.dirname(os.path.abspath(__file__))}/models/sc.h5")
    data = np.reshape(data.T, (-1, 29400, 4))
    predictions = sc_model.predict(
    [data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]]
    )
    sc_labels = ['Core','Deep','REM']
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_sc_labels = [sc_labels[label] for label in predicted_labels]
    return predicted_sc_labels


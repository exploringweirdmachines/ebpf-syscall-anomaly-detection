import argparse
import time
import pandas as pd
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from bcc import BPF
import signal

# Constants
MAX_SYSCALLS = 512
HISTOGRAM_POLL_INTERVAL = 0.1  # Poll every 100 ms
# Load eBPF code from a file called ebpf_code.c
EBPF_CODE_FILE = "ebpf_code.c"

def load_ebpf_code(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: eBPF code file '{filename}' not found.")
        exit(1)


def create_ebpf_program(target_pid):
    ebpf_code = load_ebpf_code(EBPF_CODE_FILE)

    # Set target PID in the eBPF program
    ebpf_code = ebpf_code.replace("TARGET_PID", str(target_pid))
    return BPF(text=ebpf_code)

def capture_data(b, data_file, target_pid):
    # Get the histogram map from the BPF program
    histo_map = b["histogram"]
    prev_histogram = [0] * MAX_SYSCALLS

    print(f"Capturing data from PID: {target_pid}")
    print("Press Ctrl+C to stop capturing.")

    # Setup CSV file header
    if not os.path.exists(data_file):
        with open(data_file, 'w') as f:
            f.write(','.join([f'syscall_{i}' for i in range(MAX_SYSCALLS)]) + '\n')
    
    try:
        while True:
            histogram = [histo_map[s].value for s in range(0, MAX_SYSCALLS)]
            if histogram != prev_histogram:
                deltas = [1.0 - (prev_histogram[s] / histogram[s]) if histogram[s] != 0.0 else 0.0 for s in range(0, MAX_SYSCALLS)]
                
                with open(data_file, 'a') as f:
                    f.write(','.join(map(str,deltas)) + '\n')
                prev_histogram = histogram
            time.sleep(HISTOGRAM_POLL_INTERVAL)

    except KeyboardInterrupt:
         print("\nData capturing stopped.")
    finally:
        print("Cleaning up eBPF program...")
        b.cleanup()


def build_autoencoder(n_inputs):
    inp = Input(shape=(n_inputs,))
    encoder = Dense(n_inputs)(inp)
    encoder = ReLU()(encoder)
    middle = Dense(int(n_inputs / 2))(encoder)
    decoder = Dense(n_inputs)(middle)
    decoder = ReLU()(decoder)
    decoder = Dense(n_inputs, activation='linear')(decoder)
    m = Model(inp, decoder)
    m.compile(optimizer=Adam(), loss='mse')
    return m

def train_model(data_file, model_file, epochs):
    df = pd.read_csv(data_file)
    data = df.values
    np.random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    model = build_autoencoder(MAX_SYSCALLS)
    history = model.fit(train_data, train_data, epochs=epochs, validation_data=(test_data, test_data),verbose=2)
    model.save(model_file)
    print(f"model saved to {model_file}, getting error threshold for {len(test_data)} samples ...")

    y_test = model.predict(test_data, verbose=0)
    test_err = []
    for ind in range(len(test_data)):
         abs_err = np.abs(test_data[ind, :]-y_test[ind, :])
         test_err.append(abs_err.sum())
    threshold = max(test_err)
    print(f"error threshold={threshold}")
    return threshold

def run_anomaly_detection(b, model_file, max_error, target_pid):
    print(f"Running anomaly detection on PID: {target_pid}")
    histo_map = b["histogram"]
    prev_histogram = [0] * MAX_SYSCALLS
    
    model = tf.keras.models.load_model(model_file)
    try:
        while True:
            histogram = [histo_map[s].value for s in range(0, MAX_SYSCALLS)]
            if histogram != prev_histogram:
                deltas = [1.0 - (prev_histogram[s] / histogram[s]) if histogram[s] != 0.0 else 0.0 for s in range(0, MAX_SYSCALLS)]
                current_vector = np.array(deltas).reshape(1, -1)
                reconstructed_vector = model.predict(current_vector,verbose=0)
                abs_err = np.abs(current_vector - reconstructed_vector)
                cumulative_error = abs_err.sum()
                
                if cumulative_error > max_error:
                     top_anomalies_indices = np.argsort(abs_err[0])[::-1][:3]
                     top_anomalies_names = [list(histo_map.keys())[i].decode() if isinstance(list(histo_map.keys())[i], bytes) else list(histo_map.keys())[i] for i in top_anomalies_indices]
                     top_anomalies_errors = [abs_err[0][i] for i in top_anomalies_indices]
                     print(f"error = {cumulative_error} - max = {max_error:.6f} - top 3:")
                     for i in range(3):
                        print(f"  {top_anomalies_names[i]} = {top_anomalies_errors[i]:.6f}")
                prev_histogram = histogram
            time.sleep(HISTOGRAM_POLL_INTERVAL)
    except KeyboardInterrupt:
         print("\nAnomaly detection stopped.")
    finally:
        print("Cleaning up eBPF program...")
        b.cleanup()
    


def main():
    parser = argparse.ArgumentParser(description="Process behavior anomaly detection using eBPF and autoencoders.")
    parser.add_argument("--pid", type=int, help="Target process ID.")
    parser.add_argument("--data", type=str, help="CSV file to store/load training data.", default="syscall_data.csv")
    parser.add_argument("--model", type=str, help="Path to save/load trained model.", default="syscall_model.h5")
    parser.add_argument("--learn", action="store_true", help="Capture data and store it into CSV file for learning.")
    parser.add_argument("--train", action="store_true", help="Train the autoencoder.")
    parser.add_argument("--run", action="store_true", help="Run anomaly detection on a live process")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training the autoencoder.")
    parser.add_argument("--max-error", type=float, default=10.0, help="Maximum error threshold for anomaly detection.")

    args = parser.parse_args()

    if not args.pid:
        print("Error: Please specify the target process id with --pid")
        exit(1)
   
    if not any([args.learn, args.train, args.run]):
         print("Error: Please specify an operation with --learn, --train or --run")
         exit(1)

    b = create_ebpf_program(args.pid)

    if args.learn:
      capture_data(b, args.data, args.pid)
    elif args.train:
        train_model(args.data, args.model, args.epochs)
    elif args.run:
        run_anomaly_detection(b, args.model, args.max_error, args.pid)
    

if __name__ == "__main__":
    main()

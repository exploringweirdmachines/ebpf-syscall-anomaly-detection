import argparse
import time
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from bcc import BPF
import signal


# Constants
MAX_SYSCALLS = 512
HISTOGRAM_POLL_INTERVAL = 0.1  # Poll every 100 ms
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
                    f.write(','.join(map(str, deltas)) + '\n')
                prev_histogram = histogram
            time.sleep(HISTOGRAM_POLL_INTERVAL)

    except KeyboardInterrupt:
         print("\nData capturing stopped.")
    finally:
        print("Cleaning up eBPF program...")
        b.cleanup()


class Autoencoder(nn.Module):
    def __init__(self, n_inputs):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, n_inputs),
            nn.ReLU(),
            nn.Linear(n_inputs, int(n_inputs / 2))
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(n_inputs / 2), n_inputs),
            nn.ReLU(),
            nn.Linear(n_inputs, n_inputs)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(data_file, model_file, epochs):
    df = pd.read_csv(data_file)
    data = df.values
    np.random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)


    train_dataset = TensorDataset(train_data_tensor, train_data_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(test_data_tensor, test_data_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Autoencoder(MAX_SYSCALLS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        for batch_features, _ in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, _ in test_dataloader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_features)
                val_loss += loss.item()
        val_loss /= len(test_dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.6f}')



    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}, getting error threshold for {len(test_data)} samples ...")

    model.eval()
    test_err = []
    with torch.no_grad():
        for features, _ in test_dataloader:
            outputs = model(features)
            abs_err = torch.abs(features - outputs)
            test_err.extend(torch.sum(abs_err, dim=1).tolist())

    threshold = max(test_err)
    print(f"error threshold={threshold}")
    return threshold


def run_anomaly_detection(b, model_file, max_error, target_pid):
    print(f"Running anomaly detection on PID: {target_pid}")
    histo_map = b["histogram"]
    prev_histogram = [0] * MAX_SYSCALLS
    model = Autoencoder(MAX_SYSCALLS)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    try:
        while True:
             histogram = [histo_map[s].value for s in range(0, MAX_SYSCALLS)]
             if histogram != prev_histogram:
                deltas = [1.0 - (prev_histogram[s] / histogram[s]) if histogram[s] != 0.0 else 0.0 for s in range(0, MAX_SYSCALLS)]
                current_vector = torch.tensor(deltas, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                  reconstructed_vector = model(current_vector)
                  abs_err = torch.abs(current_vector - reconstructed_vector)
                  cumulative_error = torch.sum(abs_err).item()
                  if cumulative_error > max_error:
                    top_anomalies_indices = torch.argsort(abs_err[0], descending=True)[:3]
                    top_anomalies_names = [list(histo_map.keys())[i].decode() if isinstance(list(histo_map.keys())[i], bytes) else list(histo_map.keys())[i] for i in top_anomalies_indices]
                    top_anomalies_errors = [abs_err[0][i].item() for i in top_anomalies_indices]
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
    parser.add_argument("--model", type=str, help="Path to save/load trained model.", default="syscall_model.pth")
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

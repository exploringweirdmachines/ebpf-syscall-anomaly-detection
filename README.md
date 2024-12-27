# Process Behavior Anomaly Detection using eBPF and Autoencoders

This project implements a system for detecting anomalous process behavior by monitoring system calls using eBPF and analyzing the syscall patterns using a PyTorch-based autoencoder.

## Overview

The core idea is to:

1.  **Capture System Call Data:** Use eBPF (extended Berkeley Packet Filter) to trace system calls made by a target process. Instead of tracking arguments, we track the rate of change of each syscall.
2.  **Train an Autoencoder:** Use the collected syscall data to train an autoencoder that learns what "normal" syscall patterns look like.
3.  **Detect Anomalies:** Compare the reconstruction error of the autoencoder on live process data against a learned threshold to detect anomalies.

## Dependencies

*   **Python 3.7+**
*   **PyTorch**
*   **BCC (BPF Compiler Collection)**
*   **pandas**
*   **numpy**

To install dependencies:

```bash
pip install torch torchvision torchaudio pandas numpy bcc
```

## Files

*   `main.py`: Python script containing all the logic for data capturing, model training, and anomaly detection.
*   `ebpf_code.c`: C file containing the eBPF code for system call tracing.

## Usage

### 1.  Capture Data

First, you need to capture syscall data for the process you want to monitor. This is the "learning" phase where the system learns "normal" behavior.

```bash
sudo python main.py --pid <target_pid> --data <your_data.csv> --learn
```

*   `<target_pid>`:  Replace with the process ID of the target process.
*   `<your_data.csv>`: Replace with the path to the CSV file where the captured data will be stored (e.g., `spotify.csv`).

This command will start capturing the rate of change for each syscall and store the information in the provided CSV file. Press `Ctrl+C` to stop capturing data.

### 2. Train the Autoencoder

Next, train the autoencoder using the collected data.

```bash
python main.py --data <your_data.csv> --model <your_model.pth> --train --epochs <num_epochs>
```

*   `<your_data.csv>`: The path to the CSV file created in the previous step.
*   `<your_model.pth>`: The path to save the trained model (e.g., `spotify_model.pth`).
*   `<num_epochs>`: The number of training epochs.

This command trains the autoencoder using your data, saves the trained model to a `.pth` file and prints the computed error threshold to the screen.

### 3. Run Anomaly Detection

Now you can use the trained model to detect anomalies on a live process:

```bash
sudo python main.py --pid <target_pid> --model <your_model.pth> --max-error <max_error_value> --run
```

*   `<target_pid>`: The process ID of the target process.
*   `<your_model.pth>`: The path to the trained model.
*   `<max_error_value>`: The maximum error threshold for anomaly detection. This value is automatically calculated by the training phase and printed on the screen.

This will start monitoring the target process for anomalous syscall patterns. If the calculated reconstruction error is greater than the specified threshold, the script will print out the error and the top 3 anomalous system calls. Press `Ctrl+C` to stop the process.

## `main.py` Script Details

### Data Capturing

*   Uses `bcc` to create and load an eBPF program to track system calls.
*   Polls the eBPF histogram periodically and calculates the rate of change for each syscall compared to the last sample.
*   Writes the syscall rate changes (deltas) to a CSV file for training.

### Autoencoder Training

*   Builds a simple autoencoder with an encoder, an intermediate representation layer, and a decoder using PyTorch.
*   Uses `torch.nn.MSELoss` as the loss function and `torch.optim.Adam` as the optimizer.
*   Loads training data from the CSV file and creates PyTorch `DataLoader` objects for efficient batch processing.
*   Trains the model with the training data, and prints the validation loss at the end of each epoch, using the `test_data` set in the CSV file.
*   Saves the trained model to a `.pth` file.
*   Calculates the error threshold by using the test dataset and prints the threshold value on screen.

### Anomaly Detection

*   Loads the trained model from the `.pth` file.
*   Captures live system call data using eBPF as described in the learning phase.
*   Performs a forward pass on the autoencoder using live data.
*   Calculates the reconstruction error.
*   If the error exceeds the threshold, it prints the error and the top 3 most anomalous system calls.

## eBPF Code (`ebpf_code.c`)

The eBPF code is included in `ebpf_code.c` and handles:

*   Filtering events based on the target process ID.
*   Incrementing the per-cpu histograms whenever a system call is made by the target process.

## Considerations

*   **Root Privileges:** You need root privileges to run eBPF programs, therefore the script should be run using `sudo`.
*   **BCC Installation:** Ensure you have BCC installed on your system.
*   **Tuning:** The `max_error` threshold and training hyperparameters (like learning rate, epochs, etc.) may require tuning.
*   **Syscall Identification**:  The syscalls are identified by their number, which might differ between kernel versions. This script assumes that system calls in your system have the same numbers as in the original author's environment. To solve this, you can change the eBPF code to resolve and print the names of each system call.
*   **Error Handling:** Error handling could be improved in a production environment.

## Disclaimer
### Author
This code was developed based on the ideas and methodology described in the original blog post by Simone Margaritelli <https://www.evilsocket.net/2022/08/15/Process-behaviour-anomaly-detection-using-eBPF-and-unsupervised-learning-Autoencoders/>. <br>
This is a basic implementation of a complex system. The code has been created following the specifications of the blog post using PyTorch and therefore the performance of the system will depend on several factors, such as the size and the quality of the training data.<br>
This project is mainly intended to demonstrate the described approach, so feel free to use it as inspiration for your projects!



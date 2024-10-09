# PHYSICAL Layer: Amplify And Forward relaying technique
Implement a cooperative communication system using amplify-and-forward relaying.

# Amplify-and-Forward Relaying Simulation

This repository contains the implementation of the **Amplify-and-Forward (AF) Relaying** scheme simulation, which models a cooperative communication system. The simulation incorporates Rayleigh fading, path loss, shadowing, relay selection, maximal ratio combining (MRC), and computes key performance metrics such as Bit Error Rate (BER), Signal-to-Noise Ratio (SNR), and Spectral Efficiency.

## Overview

The **Amplify-and-Forward (AF) Relaying** scheme enhances communication reliability and coverage in wireless networks. A source node transmits data to a destination node, with the help of one or more relays. These relays amplify the signal received from the source before forwarding it to the destination. 

This simulation models the transmission over a fading channel, calculates the SNR at the destination, and computes the BER using different relaying strategies. The mathematical formulation below details each step of the simulation process.

## Mathematical Model

### 1. Transmission Power Conversion

Transmission power in dB is converted to linear scale using the following equation:

\[
P_{\text{trans}} = 10^{\left(\frac{P_{\text{dB}}}{10}\right)}
\]

Where:
- \( P_{\text{trans}} \): Transmission power in Watts (linear scale).
- \( P_{\text{dB}} \): Transmission power in decibels (dB).

### 2. Channel Coefficients (Rayleigh Fading)

The channel coefficients are modeled as complex Gaussian random variables, representing Rayleigh fading:

\[
h_{sr} \sim \mathcal{CN}(0, 1), \quad h_{rd} \sim \mathcal{CN}(0, 1)
\]

- \( h_{sr} \): Channel coefficient from Source to Relay.
- \( h_{rd} \): Channel coefficient from Relay to Destination.

### 3. Path Loss and Shadowing

**Path loss** quantifies the reduction in signal power as it propagates through space:

\[
L = 10 \cdot \alpha \cdot \log_{10}(r)
\]

Where:
- \( \alpha \): Path loss exponent.
- \( r \): Distance between transmitter and receiver.

**Shadowing** adds an extra random variation to the received signal power, modeled as a Gaussian random variable:

\[
S \sim \mathcal{N}(0, \sigma_{\text{shadow}}^2)
\]

Where:
- \( \sigma_{\text{shadow}} \): Standard deviation of the shadowing effect (dB).

### 4. Modulation

Modulated symbols are mapped to a complex constellation, e.g., for QPSK modulation:

\[
s_k = e^{j \frac{2\pi k}{M}}, \quad k \in \{0, 1, \dots, M-1\}
\]

- \( s_k \): Modulated symbol.
- \( M \): Modulation order (e.g., 4 for QPSK).

### 5. Source to Relay Transmission

The signal from the source to the relay is attenuated by path loss and shadowing:

\[
y_{sr} = s \cdot 10^{-\frac{L_{sr}}{10}} \cdot 10^{\frac{S_{sr}}{10}}
\]

### 6. Relay Selection and Amplification

The best relay is selected based on channel gain:

\[
i_{\text{best}} = \arg\max_{i} |h_{sr,i}|^2
\]

The relay amplifies the received signal:

\[
y_r = y_{sr,i_{\text{best}}} \cdot \sqrt{\frac{P_{\text{trans}}}{|y_{sr,i_{\text{best}}}|^2 + 1}}
\]

### 7. Relay to Destination Transmission

The relay transmits the amplified signal to the destination, affected again by path loss and shadowing:

\[
y_{rd} = y_r \cdot 10^{-\frac{L_{rd,i_{\text{best}}}}{10}} \cdot 10^{\frac{S_{rd,i_{\text{best}}}}{10}}
\]

### 8. Maximal Ratio Combining (MRC)

The signals received directly from the source and via the relay are combined to maximize signal quality:

\[
y_{\text{combined}} = y_{sr,i_{\text{best}}} + y_{rd}
\]

### 9. SNR Calculation

The SNR at the destination for the combined signal is calculated:

\[
\text{SNR}_{\text{combined}} = \frac{|y_{\text{combined}}|^2}{1 + \text{Var}(y_{\text{combined}})}
\]

### 10. Demodulation and BER Calculation

The received signal is demodulated, and the Bit Error Rate (BER) is computed:

\[
\text{BER} = \frac{\text{Number of Error Bits}}{\text{Total Number of Bits Transmitted}}
\]

### 11. Spectral Efficiency Calculation

Spectral efficiency is measured in bits per second per Hertz (bps/Hz):

\[
\eta = \log_2(M)
\]

## Simulation Code

Below is the Python code implementing the Amplify-and-Forward relaying scheme. It simulates the transmission process, relay selection, signal amplification, and calculates the BER, SNR, and spectral efficiency.

```python
import numpy as np

# Simulation parameters
M = 4  # QPSK modulation
alpha = 3  # Path loss exponent
sigma_shadow = 8  # Shadowing std dev
P_db = 20  # Transmission power in dB
P_trans = 10**(P_db/10)  # Convert dB to linear scale
R = 2  # Number of relays
S = 10000  # Number of symbols
B = 100  # Batch size for processing

def generate_channel_coefficients(batch_size, num_relays):
    """Generates Rayleigh fading channel coefficients."""
    h_sr = np.random.normal(0, 1, (batch_size, num_relays)) + 1j * np.random.normal(0, 1, (batch_size, num_relays))
    h_rd = np.random.normal(0, 1, (batch_size, num_relays)) + 1j * np.random.normal(0, 1, (batch_size, num_relays))
    return h_sr, h_rd

def calculate_path_loss_and_shadowing(batch_size, num_relays):
    """Calculates path loss and shadowing for the source-relay and relay-destination links."""
    path_loss_sr = 10 * alpha * np.log10(np.random.uniform(0.5, 2, (batch_size, num_relays)))
    shadowing_sr = np.random.normal(0, sigma_shadow, (batch_size, num_relays))
    path_loss_rd = 10 * alpha * np.log10(np.random.uniform(0.5, 2, (batch_size, num_relays)))
    shadowing_rd = np.random.normal(0, sigma_shadow, (batch_size, num_relays))
    return path_loss_sr, shadowing_sr, path_loss_rd, shadowing_rd

def modulation(num_symbols, M):
    """Generates and modulates symbols using QPSK."""
    k = np.random.randint(0, M, num_symbols)
    s = np.exp(1j * (2 * np.pi * k / M))
    return s, k

def amplify_signal(y_sr_best, P_trans):
    """Amplifies the signal at the relay."""
    return y_sr_best * np.sqrt(P_trans / (np.abs(y_sr_best)**2 + 1))

def relay_selection(h_sr):
    """Selects the best relay based on channel gain."""
    return np.argmax(np.abs(h_sr)**2, axis=1)

def maximal_ratio_combining(y_sr_best, y_rd):
    """Combines signals received from the source and relay."""
    return y_sr_best + y_rd

def calculate_snr(y_combined):
    """Calculates the SNR of the combined signal."""
    return np.abs(y_combined)**2 / (1 + np.var(y_combined))

def demodulate_and_calculate_ber(s, y_combined, M):
    """Demodulates the combined signal and calculates the BER."""
    theta = np.angle(y_combined)
    k_hat = np.round(theta / (2 * np.pi / M)) % M
    ber = np.mean(s != k_hat)
    return ber

# Main simulation loop
ber_accumulator = []
for _ in range(S // B):
    # Channel coefficients and path loss
    h_sr, h_rd = generate_channel_coefficients(B, R)
    path_loss_sr, shadowing_sr, path_loss_rd, shadowing_rd = calculate_path_loss_and_shadowing(B, R)
    
    # Modulation
    s, k = modulation(B, M)
    
    # Source to relay transmission
    y_sr = s[:, None] * 10 ** (-path_loss_sr / 10) * 10 ** (shadowing_sr / 10)
    
    # Relay selection
    i_best = relay_selection(h_sr)
    y_sr_best = np.choose(i_best, y_sr.T)
    
    # Amplify signal
    y_r = amplify_signal(y_sr_best, P_trans)
    
    # Relay to destination transmission
    y_rd = y_r * 10 ** (-path_loss_rd[:, i_best] / 10) * 10 ** (shadowing_rd[:, i_best] / 10)
    
    # Maximal ratio combining
    y_combined = maximal_ratio_combining(y_sr_best, y_rd)
    
    # SNR calculation
    snr_combined = calculate_snr(y_combined)
    
    # BER calculation
    ber = demodulate_and_calculate_ber(k, y_combined, M)
    ber_accumulator.append(ber)

# Average BER
average_ber = np.mean(ber_accumulator)
print(f"Average BER: {average_ber:.4f}")

# PHYSICAL Layer: Amplify And Forward Relaying Technique

This repository contains the implementation of the **Amplify-and-Forward (AF) Relaying** scheme simulation, which models a cooperative communication system. The simulation incorporates Rayleigh fading, path loss, shadowing, relay selection, maximal ratio combining (MRC), and computes key performance metrics such as Bit Error Rate (BER), Signal-to-Noise Ratio (SNR), and Spectral Efficiency.

## Overview

The **Amplify-and-Forward (AF) Relaying** scheme enhances communication reliability and coverage in wireless networks. A source node transmits data to a destination node, with the help of one or more relays. These relays amplify the signal received from the source before forwarding it to the destination. 

This simulation models the transmission over a fading channel, calculates the SNR at the destination, and computes the BER using different relaying strategies. The mathematical formulation below details each step of the simulation process.

## Mathematical Model

### 1. Transmission Power Conversion

Transmission power in dB is converted to linear scale using the following equation:

P_trans = 10 ^ (P_dB / 10)

<x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

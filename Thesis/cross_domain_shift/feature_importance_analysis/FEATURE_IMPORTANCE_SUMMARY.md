# Cross-Domain Feature Importance Summary

This file summarizes feature importance behavior across outputs through outputs_5.

## IoT-23 -> UNSW-NB15

- Random Forest: flow_duration (0.476), source_to_destination_packets (0.408), protocol (0.096), connection_state (0.021), destination_to_source_packets (0.005)
- XGBoost: flow_duration (0.397), source_to_destination_packets (0.273), protocol (0.269), source_to_destination_bytes (0.023), destination_to_source_packets (0.015)

## UNSW-NB15 -> IoT-23

- Random Forest: destination_mean_bytes_per_packet (0.177), flow_duration (0.174), source_to_destination_bytes (0.144), destination_to_source_bytes (0.136), source_mean_bytes_per_packet (0.128)
- XGBoost: destination_mean_bytes_per_packet (0.361), flow_duration (0.133), destination_to_source_bytes (0.123), source_mean_bytes_per_packet (0.096), source_to_destination_bytes (0.082)

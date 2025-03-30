# WiFi Doctor: Network Performance Analysis Tool
## Computer Networks 2 - Technical University of Crete

### Project Overview
WiFi Doctor is a sophisticated network analysis tool that provides comprehensive diagnostics and performance monitoring for WiFi networks. The tool analyzes both 2.4GHz and 5GHz bands, offering detailed insights into signal quality, throughput performance, and channel utilization patterns.

### Authors
- [Sako.S, Koronaios.G, Lagos.T]
- Department of Electrical and Computer Engineering
- Technical University of Crete

### Features
1. **Signal Quality Analysis**
   - Signal strength measurements (-70 dBm threshold)
   - SNR analysis (20 dB threshold)
   - Noise level monitoring
   - Signal variation tracking

2. **Performance Metrics**
   - Throughput analysis (1-1000 Mbps range)
   - Frame retry rate monitoring
   - MCS Index analysis
   - PHY type distribution

3. **Channel Analysis**
   - 2.4GHz and 5GHz band separation
   - Channel utilization patterns
   - Interference detection
   - Band-specific performance metrics

4. **Vendor Analysis**
   - Device manufacturer distribution
   - Vendor-specific signal patterns
   - Channel usage by vendor
   - Performance comparison across vendors

### Prerequisites
#### System Requirements
- Python 3.8 or higher
- Linux-based operating system
- Network interface card supporting monitor mode
- Wireshark/TShark installed

#### Required Python Packages
```bash
# Install required packages
pip install -r requirements.txt
```

### Project Structure
WiFi_Doctor/
├── pcap_parser/
│ └── new_pcap_parser.py       PCAP file parsing functionality
├── performance_monitor/
│ ├── monitor_density.py       Network density analysis
│ └── monitor_throughput.py    Throughput analysis
├── wifi_sniffer/
│ ├── sniffer.py               Base sniffer functionality
│ └── linux_sniffer.py         Linux-specific capture
├── data/                      PCAP file storage
└── visualization_results/     Analysis output
├── density/                   Density analysis results
└── throughput/                Throughput analysis results

# RUN THE PROJECT
To be able to run the project just type: 
```bash
pyhton3 Wifi_Doctor/main.py
```
CLI output, For selecting between the Data, of the current saved pcap files, and after that choosing
the mode [density, throughtput] and the packet limit parsing.
The outputs will be stored: 
# PARSING PACKETS
```bash
/results_parser/
```
# DENSITY RESULTS
```bash
/visualization_results/density/
```
# THROUGHPUT RESULTS
```bash
/visualization_results/throughput/
```
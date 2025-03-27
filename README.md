# WiFi Doctor

A Wi-Fi network performance monitoring system designed to diagnose performance bottlenecks.

## Project Structure
- `wifi_sniffer/`: Code for capturing Wi-Fi packets.
- `pcap_parser/`: Code for parsing .pcap files.
- `performance_monitor/`: Code for analyzing Wi-Fi performance.
- `performance_analyzer/`: Code for identifying performance bottlenecks.
- `visualizer/`: Code for visualizing results.
- `data/`: Folder to store .pcap files.
- `reports/`: Folder to store the final report.

## How to Run
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install pyshark pandas matplotlib seaborn wireshark

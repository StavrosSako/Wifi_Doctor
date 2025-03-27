import os
from pcap_parser import pcap_parser
from performance_monitor import (
    channel_heatmap,
    snr_dist,
    throughput_analyzer,
    phy_summary_analyzer,
    rate_gap_analyzer,
    throughput_SNR,
    phy_data_rate
)

def run_wifi_doctor(pcap_path):
    print("[Step 1] Parsing PCAP file...")
    parsed_packets = pcap_parser.parse_pcap_file(pcap_path)
    pcap_name = os.path.basename(pcap_path).replace(".pcap", "")
    parsed_file_path = f'results_parser/{pcap_name}_parsed_packets.csv'
    pcap_parser.save_to_csv(parsed_packets, parsed_file_path)

    print("[Step 2] Generating Channel Heatmap...")
    channel_heatmap.generate(parsed_file_path)

    print("[Step 3] Generating SNR Distribution...")
    snr_dist.generate(parsed_file_path)

    print("[Step 4] Analyzing Throughput...")
    throughput_analyzer.generate(parsed_file_path)

    print("[Step 5] Creating PHY Type Summary...")
    phy_summary_analyzer.generate(parsed_file_path)

    print("[Step 6] Performing Rate Gap Analysis...")
    rate_gap_analyzer.generate(parsed_file_path)

    print("[Step 7] Correlating Throughput and SNR...")
    throughput_SNR.generate(parsed_file_path)

    print("[Step 8] PHY Data Rate Visualization...")
    phy_data_rate.generate(parsed_file_path)

    print("WiFi Doctor Analysis Complete.")

if __name__ == "__main__":
    pcap_path = '../data/HowIWiFi_PCAP.pcap'
    run_wifi_doctor(pcap_path)

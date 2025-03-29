#!/usr/bin/env python3

import argparse
import os
import subprocess

# Define default paths
default_data_dir = "./data"
default_output_dir = "./outputs"
default_results_dir = "./results_parser"
default_pcap_parser = "./pcap_parser/new_pcap_parser.py"
default_monitor_throughput = "./performance_monitor/monitor_throughput.py"

def run_wifi_doctor(pcap_path, bssid):
    print("\n🩺 Running WiFi Doctor...")

    # 1. Parse PCAP to CSV
    parsed_csv_path = os.path.join(default_results_dir, "home_parsed_packets.csv")
    print("\n🔍 Parsing PCAP file...")
    subprocess.run([
        "python3", default_pcap_parser,
        "-f", pcap_path,
        "-o", parsed_csv_path
    ])

    # 2. Analyze throughput
    print("\n📊 Analyzing throughput...")
    subprocess.run([
        "python3", default_monitor_throughput
    ])

    print("\n✅ WiFi Doctor analysis completed!")
    print(f"  → PCAP analyzed: {pcap_path}")
    print(f"  → BSSID used: {bssid}")
    print(f"  → Results saved to: {default_output_dir}/throughput_results")


def main():
    parser = argparse.ArgumentParser(description="WiFi Doctor - Analyze your WiFi network from PCAP files")
    parser.add_argument("--pcap", type=str, help="Path to the PCAP file", required=True)
    parser.add_argument("--bssid", type=str, help="Access Point BSSID", required=True)
    args = parser.parse_args()

    run_wifi_doctor(args.pcap, args.bssid)


if __name__ == "__main__":
    main()

import pyshark
import re
import csv
import os

def parse_pcap_file(pcap_path, packet_limit=200):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    field_mapping = {
        'bss id': 'BSS Id',
        'receiver address': 'Receiver address',
        'type/subtype': 'Type/Subtype',
        'phy type': 'PHY type',
        'mcs index': 'MCS Index',
        'bandwidth': 'Bandwidth',
        'short gi': 'Short GI',
        'data rate': 'Data rate',
        'channel': 'Channel',
        'frequency': 'Frequency',
        'signal strength': 'Signal strength',
        'signal/noise ratio': 'Signal/noise ratio',
        'timestamp': 'TSF Timestamp'
    }

    results = []

    with pyshark.FileCapture(pcap_path) as capture:
        for packet_count, packet in enumerate(capture):
            if packet_count >= packet_limit:
                break

            packet_info = {field_name: "N/A" for field_name in field_mapping.values()}
            packet_str = ansi_escape.sub('', str(packet))
            lines = packet_str.split('\n')

            # Store all antenna signal values with line indices
            signal_map = {}
            antenna_signals = {}

            for i, line in enumerate(lines):
                lower = line.lower()

                # Extract fields normally
                for key, field_name in field_mapping.items():
                    if key in lower:
                        parts = re.split(r':\s*', line, maxsplit=1)
                        if len(parts) > 1:
                            packet_info[field_name] = parts[1].strip()

                # Frequency
                if "frequency" in lower or "mhz" in lower:
                    match = re.search(r'(\d{4})MHz', line)
                    if match:
                        packet_info["Frequency"] = match.group(1)

                # Data Rate
                if "data rate" in lower:
                    match = re.search(r'(\d+\.?\d*) Mb/s', line)
                    if match:
                        packet_info["Data rate"] = match.group(1) + " Mbps"

                # Track all antenna signals with their line numbers
                if "antenna signal" in lower:
                    match = re.search(r'antenna signal:\s*(-?\d+)\s*dBm', line)
                    if match:
                        signal_map[i] = int(match.group(1))

                # Match antenna index with previous antenna signal
                if "antenna: 0" in lower:
                    # Find the nearest preceding antenna signal
                    nearest_signal = next((signal_map[j] for j in reversed(range(i)) if j in signal_map), None)
                    if nearest_signal is not None:
                        antenna_signals[0] = nearest_signal

                if "antenna: 1" in lower:
                    nearest_signal = next((signal_map[j] for j in reversed(range(i)) if j in signal_map), None)
                    if nearest_signal is not None:
                        antenna_signals[1] = nearest_signal

                        # ➕ Calculate SNR if signal strength is available
            signal_dbm = None
            if packet_info["Signal strength"] != "N/A":
                match = re.search(r'(-?\d+)\s*dBm', packet_info["Signal strength"])
                if match:
                        signal_dbm = int(match.group(1))
                        assumed_noise = -95  # Assumed noise floor for 2.4 GHz
                        snr = signal_dbm - assumed_noise
                        packet_info["Signal/noise ratio"] = f"{snr} dB"
                else:
                    packet_info["Signal/noise ratio"] = "N/A"
            else:
                packet_info["Signal/noise ratio"] = "N/A"


            results.append(packet_info)

    return results


def save_to_csv(parsed_packets, output_file):
    if not parsed_packets:
        print("No data to save.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = parsed_packets[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(parsed_packets)

    print(f"Data saved to {output_file}")


if __name__ == '__main__':
    pcap_path = '../data/HowIWiFi_PCAP.pcap'
    pcap_name = os.path.basename(pcap_path).replace('.pcap', '')
    parsed_packets = parse_pcap_file(pcap_path, packet_limit=151)
    print(f"\nFound {len(parsed_packets)} packets.\n")

    output_file = f'../results_parser/{pcap_name}_parsed_packets.csv'
    save_to_csv(parsed_packets, output_file)

import pyshark
import re
import csv
import os

def parse_pcap_file(pcap_path, packet_limit=500):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    field_mapping = {
        'bss id': 'BSS Id',
        'receiver address': 'Receiver address',
        'type/subtype': 'Type/Subtype',
        'phy type': 'PHY type',
        'mcs index': 'MCS Index',
        'bandwidth': 'Bandwidth',
        'channel width': 'Bandwidth',
        'short gi': 'Short GI',
        'data rate': 'Data rate',
        'channel': 'Channel',
        'frequency': 'Frequency',
        'signal strength': 'Signal strength',
        'signal/noise ratio': 'Signal/noise ratio',
        'retry': 'Retry',
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

            for i, line in enumerate(lines):
                lower = line.lower()

                # Extract general fields
                for key, field_name in field_mapping.items():
                    if key in lower and field_name not in ['Frequency', 'Data rate', 'Bandwidth']:
                        parts = re.split(r':\s*', line, maxsplit=1)
                        if len(parts) > 1:
                            packet_info[field_name] = parts[1].strip()

                # Frequency (in MHz)
                if "frequency" in lower or "mhz" in lower:
                    match = re.search(r'(\d{4})MHz', line)
                    if match:
                        packet_info["Frequency"] = match.group(1)

                # Data Rate (in Mbps)
                if "data rate" in lower:
                    match = re.search(r'(\d+\.?\d*)\s*Mb/s', line)
                    if match:
                        packet_info["Data rate"] = match.group(1) + " Mbps"

                # Bandwidth from IEEE 802.11 Radio Info
                if "bandwidth:" in lower and "mhz" in lower:
                    match = re.search(r'bandwidth:\s*(\d+)\s*MHz', line, re.IGNORECASE)
                    if match:
                        packet_info["Bandwidth"] = f"{match.group(1)} MHz"
                elif "channel width:" in lower and "mhz" in lower:
                    match = re.search(r'channel width:\s*(\d+)\s*MHz', line, re.IGNORECASE)
                    if match:
                        packet_info["Bandwidth"] = f"{match.group(1)} MHz"
                #Retry.
                if "Retry: Set" in packet_str or "Retry: 1" in packet_str:
                    packet_info["Retry"] = 1
                else:
                    packet_info["Retry"] = 0   



            # Calculate SNR using signal strength (if present)
            signal_dbm = None
            if packet_info["Signal strength"] != "N/A":
                match = re.search(r'(-?\d+)\s*dBm', packet_info["Signal strength"])
                if match:
                    signal_dbm = int(match.group(1))
                    assumed_noise = -95
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
    pcap_path = '../data/home.pcap'
    pcap_name = os.path.basename(pcap_path).replace('.pcap', '')
    parsed_packets = parse_pcap_file(pcap_path, packet_limit=12000)
    print(f"\nFound {len(parsed_packets)} packets.\n")

    output_file = f'../results_parser/{pcap_name}_parsed_packets.csv'
    save_to_csv(parsed_packets, output_file)

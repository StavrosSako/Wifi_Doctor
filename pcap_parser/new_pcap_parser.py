import pyshark
import re
import csv


def parse_pcap_file(pcap_path, packet_limit=200): 
    """Parses a pcap file and extracts the relevant informations from the Packets"""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m') #Removing the ansi escape sequences from the packet string if there are any. 

    field_mapping = {
        'BSS Id': 'BSS Id',
        'Receiver address': 'Receiver address',
        'Type/Subtype': 'Type/Subtype',
        'PHY type': 'PHY type',
        'MCS Index': 'MCS Index',
        'Bandwidth': 'Bandwidth',
        '0... .... = Short GI': 'Short GI',
        'Data rate': 'Data rate',
        'Channel': 'Channel',
        'Frequency': 'Frequency',
        'Signal strength': 'Signal strength',
        'Timestamp': 'TSF Timestamp'
    }

    results = []

    #now we open the pcap file and read it
    with pyshark.FileCapture(pcap_path) as capture:
        for packet_count, packet in enumerate(capture):
            if packet_count >= packet_limit:
                break

            packet_info = {}
            packet_str = ansi_escape.sub('', str(packet))   #Removing the ansi escape sequences from the packet string if there are any. 

            #Extracting the relevant informations from the packet
            for line in packet_str.split('\n'):
                for key, field_name in field_mapping.items():
                    if key in line: 
                        clean_line = line.replace(':\t', '')
                        parts = clean_line.split(':', 1)
                        packet_info[field_name] = parts[1].strip() if len(parts) > 1 else clean_line.strip()

            results.append(packet_info)

    return results

def save_to_csv(data, output_file):
    """Saves the parsed data to a CSV file"""
    if not parsed_packets:
        print("No data to save.")
        return
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = parsed_packets[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(parsed_packets)
    print(f"Data saved to {output_file}")
            

if __name__ == '__main__':
    pcap_path = '/Users/user/Desktop/σχολη/8o εξάμηνο/Δίκτυα 2/Project1/Wifi_Doctor/data/HowIWiFi_PCAP.pcap'
    parsed_packets = parse_pcap_file(pcap_path, packet_limit=151)
    print(f"\nWe found {len(parsed_packets)} packets.\n")
    for i, pkt in enumerate(parsed_packets):
        print(f"Packet {i+1}:")
    
    output_file = '/Users/user/Desktop/σχολη/8o εξάμηνο/Δίκτυα 2/Project1/Wifi_Doctor/results/parsed_packets.csv'
    save_to_csv(parsed_packets, output_file)    

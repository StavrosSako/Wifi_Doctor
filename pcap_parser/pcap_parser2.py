import pyshark
import re

def parse_pcap_file(pcap_path, packet_limit=200):
    capture = pyshark.FileCapture(pcap_path)
    packet_count = 0
    results = []
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    for packets in capture:
        if packet_count >= packet_limit:
            break
        packet_count += 1
        pkt_info = {}
        packet_str = str(packets)
        packet_str = ansi_escape.sub('', packet_str)

        for line in packet_str.split('\n'):
            if 'BSS Id' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['BSS Id'] = parts[1].strip()
                else:
                    pkt_info['BSS Id'] = clean_line.strip()

            if 'Receiver address' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Receiver address'] = parts[1].strip()
                else:
                    pkt_info['Receiver address'] = clean_line.strip()

            if 'Type/Subtype' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Type/Subtype'] = parts[1].strip()
                else:
                    pkt_info['Type/Subtype'] = clean_line.strip()

            if 'PHY type' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['PHY type'] = parts[1].strip()
                else:
                    pkt_info['PHY type'] = clean_line.strip()

            if 'MCS Index' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['MCS Index'] = parts[1].strip()
                else:
                    pkt_info['MCS Index'] = clean_line.strip()

            if 'Bandwidth' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Bandwidth'] = parts[1].strip()
                else:
                    pkt_info['Bandwidth'] = clean_line.strip()

            if '0... .... = Short GI' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Short GI'] = parts[1].strip()
                else:
                    pkt_info['Short GI'] = clean_line.strip()

            if 'Data rate' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Data rate'] = parts[1].strip()
                else:
                    pkt_info['Data rate'] = clean_line.strip()

            if 'Channel' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Channel'] = parts[1].strip()
                else:
                    pkt_info['Channel'] = clean_line.strip()

            if 'Frequency' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Frequency'] = parts[1].strip()
                else:
                    pkt_info['Frequency'] = clean_line.strip()

            if 'Signal strength' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['Signal strength'] = parts[1].strip()
                else:
                    pkt_info['Signal strength'] = clean_line.strip()

            if 'Timestamp:' in line:
                clean_line = line.replace(':\t', '')
                parts = clean_line.split(':', 1)
                if len(parts) > 1:
                    pkt_info['TSF Timestamp'] = parts[1].strip()
                else:
                    pkt_info['TSF Timestamp'] = clean_line.strip()

        results.append(pkt_info)

    capture.close()
    return results

if __name__ == '__main__':
    pcap_file = '/Users/user/Desktop/σχολη/8o εξάμηνο/Δίκτυα 2/Project1/Wifi_Doctor/data/HowIWiFi_PCAP.pcap'
    parsed_packets = parse_pcap_file(pcap_file, packet_limit=151)
    print(f"\nWe found {len(parsed_packets)} packets.\n")
    for i, pkt in enumerate(parsed_packets):
        print(f"Frame {i+1}: {pkt}")

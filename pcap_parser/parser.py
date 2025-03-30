import argparse
import pyshark

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse PCAP file for WiFi analysis')
    parser.add_argument('pcap_file', help='Path to the PCAP file')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of packets to parse (0 for all packets)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Open the capture file
    capture = pyshark.FileCapture(args.pcap_file, display_filter='wlan')
    
    # Initialize packet counter
    packet_count = 0
    
    # Process packets
    for packet in capture:
        if args.limit > 0 and packet_count >= args.limit:
            break
            
        # Your existing packet processing code here
        packet_count += 1
    
    # Close the capture
    capture.close()

if __name__ == '__main__':
    main() 
import subprocess
import os

def capture_packets(interface, output_file, packet_count=1000):
    """
    Capture Wi-Fi packets using tshark and save them to a .pcap file.

    Args:
        interface (str): The Wi-Fi interface to capture packets from.
        output_file (str): The name of the output .pcap file.
        packet_count (int): The number of packets to capture.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Command to capture packets using tshark
        command = [
            "tshark",
            "-i", interface,  # Wi-Fi interface
            "-f", "type mgt subtype beacon",  # Capture only beacon frames
            "-w", output_file,  # Output .pcap file
            "-c", str(packet_count)  # Number of packets to capture
        ]

        # Run the command
        print(f"Capturing {packet_count} packets on interface {interface}...")
        subprocess.run(command, check=True)
        print(f"Capture complete. Saved to {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    interface = "en0"
    output_file = "../data/home_network_2.4GHz.pcap"  # Save to the data folder
    packet_count = 1000  # Number of packets to capture

    # Start capturing packets
    capture_packets(interface, output_file, packet_count)
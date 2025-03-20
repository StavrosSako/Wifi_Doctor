import subprocess
import os
import platform
import time

def enable_monitor_mode(interface): 
    system = platform.system() 
    
    if system == "Darwin":
        print(f"[*] Putting {interface} into monitor mode on macOS..")
        try:
            #using airport to sniff on a specifing channel
            subprocess.run(["sudo", "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
                          interface, "sniff", "1"], check=True)
            print(f"[+] {interface} is now on monitor mode.")
            return True
        except subprocess.CalledProcessError as e: 
            print(f"[-] Failled to enable monitor mode on macOS: {e}")
            return False

    elif system == "Linux":
        print(f"[*] Putting {interface} into monitor mode on Linux..")
        try:
            # Use airmon-ng to enable monitor mode
            subprocess.run(["sudo", "airmon-ng", "start", interface], check=True)
            print(f"[+] {interface} is now in monitor mode.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[-] Failed to enable monitor mode on Linux: {e}")
            return False

    else:
        print(f"[-] Unsupported operating system: {system}")
        return False

def disable_monitor_mode(interface):
    """
    Disable monitor mode and restore the Wi-Fi interface to its original state.
    Supports macOS and Linux.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        print(f"[*] Restoring {interface} to managed mode on macOS...")
        try:
            # Stop the sniffing process
            subprocess.run(["sudo", "pkill", "airport"], check=True)
            print(f"[+] {interface} is now back in managed mode.")
        except subprocess.CalledProcessError as e:
            print(f"[-] Failed to disable monitor mode on macOS: {e}")

    elif system == "Linux":  # Linux
        print(f"[*] Restoring {interface} to managed mode on Linux...")
        try:
            # Use airmon-ng to disable monitor mode
            subprocess.run(["sudo", "airmon-ng", "stop", interface], check=True)
            print(f"[+] {interface} is now back in managed mode.")
        except subprocess.CalledProcessError as e:
            print(f"[-] Failed to disable monitor mode on Linux: {e}")

def capture_packets(interface, output_file, packet_count=1000):
    """
    Capture Wi-Fi packets using tshark and save them to a .pcap file.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Command to capture packets using tshark
        command = [
            "tshark",
            "-i", interface,  # Wi-Fi interface
            "-w", output_file,  # Output .pcap file
            "-c", str(packet_count)  # Number of packets to capture
        ]

        # Run the command
        print(f"[*] Capturing {packet_count} packets on interface {interface}...")
        subprocess.run(command, check=True)
        print(f"[+] Capture complete. Saved to {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"[-] Error: {e}")
    except Exception as e:
        print(f"[-] An error occurred: {e}")

if __name__ == "__main__":
    # Replace 'en0' (macOS) or 'wlan0' (Linux) with your Wi-Fi interface name
    interface = "en0" if platform.system() == "Darwin" else "wlan0"
    output_file = "../data/home_network_2.4GHz.pcap"  # Save to the data folder
    packet_count = 1000  # Number of packets to capture

    # Enable monitor mode
    if enable_monitor_mode(interface):
        # Capture packets
        capture_packets(interface, output_file, packet_count)

        # Disable monitor mode
        #testing
        disable_monitor_mode(interface)
    else:
        print("[-] Monitor mode could not be enabled. Exiting.")

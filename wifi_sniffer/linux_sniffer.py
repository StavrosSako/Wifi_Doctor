import subprocess
import os

def run_airmon_check_kill():
    print("[*] Killing interfering processes (airmon-ng check kill)...")
    try:
        subprocess.run(["sudo", "airmon-ng", "check", "kill"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to run airmon-ng check kill: {e}")

def restore_network_manager():
    print("[*] Restarting NetworkManager service...")
    try:
        subprocess.run(["sudo", "service", "NetworkManager", "start"], check=True)
        print("[+] NetworkManager restarted.")
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to restart NetworkManager: {e}")

def enable_monitor_mode(interface):
    print(f"[*] Putting {interface} into monitor mode on Linux...")
    try:
        subprocess.run(["sudo", "airmon-ng", "start", interface], check=True)
        monitor_interface = interface + "mon"
        print(f"[+] {monitor_interface} is now in monitor mode.")
        return monitor_interface
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to enable monitor mode on Linux: {e}")
        return None

def set_channel(interface, channel):
    try:
        subprocess.run(["sudo", "iwconfig", interface, "channel", str(channel)], check=True)
        print(f"[+] {interface} set to channel {channel}.")
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to set {interface} to channel {channel}: {e}")

def disable_monitor_mode(monitor_interface):
    print(f"[*] Restoring {monitor_interface} to managed mode on Linux...")
    try:
        subprocess.run(["sudo", "airmon-ng", "stop", monitor_interface], check=True)
        print(f"[+] {monitor_interface} is now back in managed mode.")
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to disable monitor mode on Linux: {e}")

def capture_packets(interface, output_filename, packet_count=1000):
    try:
        output_path = os.path.join("/tmp", output_filename)

        # Check if file exists and delete it
        if os.path.exists(output_path):
            print(f"[-] File {output_path} already exists. Deleting...")
            os.remove(output_path)

        command = [
            "tshark",
            "-i", interface,
            "-w", output_path,
            "-c", str(packet_count)
        ]
        print(f"[*] Capturing {packet_count} packets on interface {interface}...")
        subprocess.run(command, check=True)
        print(f"[+] Capture complete. Saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[-] Error during capture: {e}")
    except Exception as e:
        print(f"[-] An error occurred: {e}")


def prompt_for_capture(label):
    print(f"\n=== Starting capture for {label} ===")
    
    interface = input("Enter base wireless interface (e.g. wlo1): ").strip()
    channel = input(f"Enter Wi-Fi channel for {label} : ").strip()
    packet_count = input("Enter number of packets to capture: ").strip()

    try:
        channel = int(channel)
        packet_count = int(packet_count)
    except ValueError:
        print("[-] Invalid input. Channel and packet count must be integers.")
        return

    output_filename = f"capture_{label.replace(' ', '_')}.pcap"

    monitor_interface = enable_monitor_mode(interface)
    if monitor_interface:
        set_channel(monitor_interface, channel)
        capture_packets(monitor_interface, output_filename, packet_count)
        disable_monitor_mode(monitor_interface)
    else:
        print(f"[-] Monitor mode couldn't be enabled for {label} capture. Skipping...")

def prompt_user_choice():
    print("\nSelect capture mode:")
    print("1. 2.4 GHz only")
    print("2. 5 GHz only")
    print("3. Both 2.4 GHz and 5 GHz")
    choice = input("Enter your choice: ").strip()
    return choice

if __name__ == "__main__":
    run_airmon_check_kill()
    
    choice = prompt_user_choice()
    if choice == "1":
        prompt_for_capture("2.4 GHz")
    elif choice == "2":
        prompt_for_capture("5 GHz")
    elif choice == "3":
        prompt_for_capture("2.4 GHz")
        prompt_for_capture("5 GHz")
    else:
        print("[-] Invalid choice. Exiting.")

    restore_network_manager()

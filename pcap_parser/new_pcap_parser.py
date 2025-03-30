import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path
import os
from datetime import datetime
import pyshark
from tqdm import tqdm

def freq_to_channel(freq):
    """Convert frequency to channel number."""
    if 2412 <= freq <= 2484:  # 2.4 GHz
        if freq == 2484:
            return 14
        return (freq - 2412) // 5 + 1
    elif 5170 <= freq <= 5825:  # 5 GHz
        return (freq - 5170) // 5 + 34
    return None

def generate_freq_to_channel_map():
    """Generate a mapping of frequencies to channels for both 2.4GHz and 5GHz bands."""
    freq_to_channel = {}
    
    # 2.4 GHz channels (1-14)
    for channel in range(1, 15):
        freq = 2407 + (channel * 5)
        freq_to_channel[freq] = channel
    
    # 5 GHz channels
    # UNII-1 (36-48)
    for channel in range(36, 49, 4):
        freq = 5000 + (channel * 5)
        freq_to_channel[freq] = channel
    
    # UNII-2 (52-64)
    for channel in range(52, 65, 4):
        freq = 5000 + (channel * 5)
        freq_to_channel[freq] = channel
    
    # UNII-2 Extended (100-144)
    for channel in range(100, 145, 4):
        freq = 5000 + (channel * 5)
        freq_to_channel[freq] = channel
    
    # UNII-3 (149-165)
    for channel in range(149, 166, 4):
        freq = 5000 + (channel * 5)
        freq_to_channel[freq] = channel
    
    return freq_to_channel

def extract_data_rate(data_rate):
    """Extract data rate from string."""
    if data_rate == 'N/A':
        return 'N/A'
    try:
        # Try to convert directly first
        return float(data_rate)
    except ValueError:
        # If direct conversion fails, try to extract from string
        match = re.search(r'(\d+(?:\.\d+)?)', str(data_rate))
        if match:
            return float(match.group(1))
        return 'N/A'

def extract_signal_strength(signal_str):
    """Extract numeric signal strength from string."""
    if not signal_str or signal_str == 'N/A':
        return None
    try:
        # Extract numeric value from signal string (e.g., "-65 dBm" -> -65)
        match = re.search(r'(-?\d+)', signal_str)
        if match:
            return float(match.group(1))
        return None
    except:
        return None

def extract_noise_level(noise_str):
    """Extract numeric noise level from string."""
    if not noise_str or noise_str == 'N/A':
        return None
    try:
        # Extract numeric value from noise string (e.g., "-92 dBm" -> -92)
        match = re.search(r'(-?\d+)', noise_str)
        if match:
            return float(match.group(1))
        return None
    except:
        return None

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio."""
    if signal is None or noise is None:
        return None
    return signal - noise

def extract_packet_info(packet, debug=False):
    try:
        # Extract signal strength
        if hasattr(packet, 'radiotap'):
            signal_strength = float(packet.radiotap.dbm_antsignal) if hasattr(packet.radiotap, 'dbm_antsignal') else float(packet.radiotap.signal_dbm) if hasattr(packet.radiotap, 'signal_dbm') else 'N/A'
            
            # Extract noise level - check multiple possible field names
            try:
                if hasattr(packet.radiotap, 'dbm_antnoise'):
                    noise_level = float(packet.radiotap.dbm_antnoise)
                elif hasattr(packet.radiotap, 'noise_dbm'):
                    noise_level = float(packet.radiotap.noise_dbm)
                else:
                    # Estimate noise level based on frequency
                    if hasattr(packet.radiotap, 'channel_freq'):
                        freq = float(packet.radiotap.channel_freq)
                        if 2400 <= freq <= 2500:  # 2.4 GHz band
                            noise_level = -95  # Typical noise floor for 2.4 GHz
                        elif 5000 <= freq <= 6000:  # 5 GHz band
                            noise_level = -98  # Typical noise floor for 5 GHz
                        else:
                            noise_level = -95  # Default to 2.4 GHz value
                    else:
                        noise_level = -95  # Default to 2.4 GHz value
            except Exception as e:
                if debug:
                    print(f"Error extracting noise level: {e}")
                noise_level = -95  # Default to 2.4 GHz value

            # Calculate SNR
            if isinstance(signal_strength, (int, float)):
                snr = signal_strength - noise_level
            else:
                snr = 'N/A'
        else:
            signal_strength = 'N/A'
            noise_level = 'N/A'
            snr = 'N/A'

        if debug:
            print(f"Signal strength: {signal_strength} dBm")
            print(f"Noise level: {noise_level} dBm")
            print(f"SNR: {snr} dB")
    except Exception as e:
        if debug:
            print(f"Error extracting packet information: {e}")
        signal_strength = 'N/A'
        noise_level = 'N/A'
        snr = 'N/A'

    return signal_strength, noise_level, snr

def parse_pcap_file(file_path, output_file, limit=None, debug=False, mode='density'):
    """Parse PCAP file and extract relevant information.
    
    Args:
        file_path: Path to the PCAP file
        output_file: Path to save the output CSV
        limit: Maximum number of packets to process
        debug: Enable debug mode
        mode: 'density' for beacon frame analysis or 'throughput' for data frame analysis
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set display filter based on mode
    if mode == 'density':
        display_filter = "wlan.fc.type_subtype == 8"  # Beacon frames
    else:  # throughput mode
        display_filter = "wlan.fc.type == 2 && wlan_radio"  # Data frames with radio information
    
    # Initialize capture with debug mode and display filter
    capture = pyshark.FileCapture(file_path, display_filter=display_filter, debug=debug)
    
    # Initialize lists to store packet information
    packets = []
    
    # Set default limit to 5000 if not specified
    if limit is None:
        limit = 5000
    
    # Get total number of packets for progress bar
    total_packets = len(capture)
    total_packets = min(total_packets, limit)
    
    # Create progress bar with more information
    pbar = tqdm(total=total_packets, 
                desc="Processing packets",
                unit="pkt",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    try:
        for packet in capture:
            if len(packets) >= limit:
                break
                
            try:
                # Check if packet has WLAN layer
                if not hasattr(packet, 'wlan'):
                    continue
                
                # Extract basic information
                frame_type = getattr(packet.wlan, 'fc_type_subtype', 'N/A')
                bss_id = getattr(packet.wlan, 'bssid', 'N/A')
                transmitter = getattr(packet.wlan, 'ta', 'N/A')
                receiver = getattr(packet.wlan, 'ra', 'N/A')
                
                # Extract PHY information with better error handling
                try:
                    phy_type = getattr(packet.wlan_radio, 'phy', 'N/A')
                    data_rate = getattr(packet.wlan_radio, 'data_rate', 'N/A')
                    
                    # Extract MCS, bandwidth, and Short GI from radiotap layer
                    if hasattr(packet, 'radiotap'):
                        # Extract MCS index
                        if hasattr(packet.radiotap, 'mcs'):
                            mcs_index = getattr(packet.radiotap, 'mcs', 'N/A')
                        elif hasattr(packet.radiotap, 'he_data_3_data_mcs'):
                            mcs_index = getattr(packet.radiotap, 'he_data_3_data_mcs', 'N/A')
                        else:
                            mcs_index = 'N/A'
                            
                        # Extract bandwidth
                        if hasattr(packet.radiotap, 'channel_flags'):
                            flags = packet.radiotap.channel_flags
                            if hasattr(flags, 'quarter'):
                                bandwidth = 5
                            elif hasattr(flags, 'half'):
                                bandwidth = 10
                            elif hasattr(flags, 'turbo'):
                                bandwidth = 40
                            else:
                                bandwidth = 20
                        elif hasattr(packet.radiotap, 'he_data_5_data_bw_ru_allocation'):
                            bw_ru = getattr(packet.radiotap, 'he_data_5_data_bw_ru_allocation', 'N/A')
                            if bw_ru != 'N/A':
                                if bw_ru == '0':
                                    bandwidth = 20
                                elif bw_ru == '1':
                                    bandwidth = 40
                                elif bw_ru == '2':
                                    bandwidth = 80
                                elif bw_ru == '3':
                                    bandwidth = 160
                                else:
                                    bandwidth = 'N/A'
                        else:
                            bandwidth = 'N/A'
                            
                        # Extract Short GI
                        if hasattr(packet.radiotap, 'flags_shortgi'):
                            short_gi = getattr(packet.radiotap, 'flags_shortgi', '0')
                            short_gi = short_gi.lower() == 'true' if isinstance(short_gi, str) else bool(int(short_gi))
                        elif hasattr(packet.radiotap, 'he_data_5_gi'):
                            gi = getattr(packet.radiotap, 'he_data_5_gi', 'N/A')
                            if gi != 'N/A':
                                short_gi = gi in ['1', '2']  # GI values 1 and 2 represent short GI
                            else:
                                short_gi = 'N/A'
                        else:
                            short_gi = 'N/A'
                    else:
                        mcs_index = 'N/A'
                        bandwidth = 'N/A'
                        short_gi = 'N/A'
                    
                except Exception as e:
                    if debug:
                        print(f"\nError extracting PHY information: {str(e)}")
                    phy_type = 'N/A'
                    mcs_index = 'N/A'
                    bandwidth = 'N/A'
                    short_gi = 'N/A'
                    data_rate = 'N/A'
                
                # Extract signal information with better error handling
                try:
                    signal_strength, noise_level, snr = extract_packet_info(packet, debug)
                except Exception as e:
                    if debug:
                        print(f"Error extracting signal information: {e}")
                    signal_strength = 'N/A'
                    noise_level = 'N/A'
                    snr = 'N/A'
                
                # Extract retry information
                retry = getattr(packet.wlan, 'fc_retry', '0')
                
                # Extract channel information
                actual_channel = getattr(packet.wlan_radio, 'channel', 'N/A')
                
                # Convert values to appropriate types
                data_rate = extract_data_rate(data_rate)
                
                # Store packet information
                packets.append({
                    'BSS Id': bss_id,
                    'Receiver address': receiver,
                    'Transmitter address': transmitter,
                    'Type/Subtype': frame_type,
                    'PHY type': phy_type,
                    'MCS Index': mcs_index,
                    'Bandwidth': bandwidth,
                    'Short GI': short_gi,
                    'Data Rate': data_rate,
                    'Signal strength': signal_strength,
                    'Noise level': noise_level,
                    'Signal/noise ratio': snr,
                    'Actual Channel': actual_channel,
                    'Retry': retry
                })
                
                # Update progress bar
                pbar.update(1)
                
            except Exception as e:
                if debug:
                    print(f"\nError processing packet: {str(e)}")
                continue
                
    except Exception as e:
        if debug:
            print(f"Error during capture: {str(e)}")
    finally:
        pbar.close()
        capture.close()
    
    # Create DataFrame and save to CSV
    if packets:
        df = pd.DataFrame(packets)
        # Reorder columns to ensure consistent output
        columns = [
            'BSS Id', 'Receiver address', 'Transmitter address', 'Type/Subtype',
            'PHY type', 'MCS Index', 'Bandwidth', 'Short GI', 'Data Rate',
            'Signal strength', 'Noise level', 'Signal/noise ratio',
            'Actual Channel', 'Retry'
        ]
        df = df[columns]
        
        # Check if output file exists
        if os.path.exists(output_file):
            # Read existing CSV
            existing_df = pd.read_csv(output_file)
            # Append new data
            df = pd.concat([existing_df, df], ignore_index=True)
            # Save the concatenated DataFrame
            df.to_csv(output_file, index=False)
            print(f"\nAppended {len(packets)} new packets to existing file")
        else:
            # Create new file
            df.to_csv(output_file, index=False)
            print(f"\nCreated new file with {len(packets)} packets")
        
        print(f"Results saved to {output_file}")
    else:
        print("No relevant packets found")

def main():
    parser = argparse.ArgumentParser(description='Parse PCAP file and extract WiFi information')
    parser.add_argument('-f', '--file', required=True, help='Input PCAP file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-l', '--limit', type=int, default=5000, help='Limit number of packets to process (default: 5000)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-m', '--mode', choices=['density', 'throughput'], default='density',
                      help='Analysis mode: density (beacon frames) or throughput (data frames)')
    
    args = parser.parse_args()
    parse_pcap_file(args.file, args.output, args.limit, args.debug, args.mode)

if __name__ == "__main__":
    main()
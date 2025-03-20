import pyshark

def parse_pcap_file(pcap_path, packet_limit=200):
    capture = pyshark.FileCapture(pcap_path)

    packet_count = 0
    results = []
    for packet in capture:
        if packet_count >= packet_limit:
            break
        packet_count += 1
        pkt_info = {
            'BSSID': None,
            'Transmitter MAC address': None,
            'Receiver MAC address': None,
            'Type/subtype': None,
            'PHY Type': None,
            'MCS Index': None,
            'Bandwidth': None,
            'Spatial Streams': None,
            'Short GI': None,
            'Data rate': None,
            'Channel': None,
            'Frequency': None,
            'Signal strength': None,
            'Signal/noise ratio': None,
            'TSF Timestamp': None
        }

        for layer in packet.layers:
            layer_name = layer.layer_name.lower()

            # WLAN Layer
            if layer_name == 'wlan':
                pkt_info['BSSID'] = layer.get_field('bssid')
                pkt_info['Transmitter MAC address'] = layer.get_field('ta')
                pkt_info['Receiver MAC address'] = layer.get_field('ra')
                pkt_info['Type/subtype'] = layer.get_field('fc_type_subtype')

            # WLAN_RADIO Layer
            if layer_name == 'wlan_radio':
                pkt_info['PHY Type'] = layer.get_field('phy')
                pkt_info['Data rate'] = f"{layer.get_field('data_rate')} Mbps"
                pkt_info['Channel'] = layer.get_field('channel')
                pkt_info['Frequency'] = f"{layer.get_field('frequency')} MHz"
                pkt_info['Signal strength'] = f"{layer.get_field('signal_dbm')} dBm"
                pkt_info['Signal/noise ratio'] = f"{layer.get_field('snr')} dB"

            # RADIOTAP Layer 
            if layer_name == 'radiotap':
                pkt_info['Short GI'] = layer.get_field('flags_shortgi')
                pkt_info['MCS Index'] = layer.get_field('mcs_index')
                pkt_info['Bandwidth'] = layer.get_field('mcs_bw')

            # WLAN_MGT Layer 
            if "mgt" in layer_name:
                pkt_info['TSF Timestamp'] = layer.get_field('wlan_fixed_timestamp')

        results.append(pkt_info)

    capture.close()
    return results


if __name__ == '__main__':
    pcap_file = 'HowIWiFi_PCAP.pcap'
    parsed_packets = parse_pcap_file(pcap_file, packet_limit=200)

    print(f"\nWe found {len(parsed_packets)} packets.\n")
    for i, pkt in enumerate(parsed_packets):  
        print(f"Frame {i+1}: {pkt}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def is_unicast_mac(mac):
    mac = mac.lower()
    return not (
        mac.startswith('ff:') or
        mac.startswith('33:33:') or
        mac.startswith('01:00:5e')
    )


def analyze_ap_clients(csv_path, ap_bssid, output_dir="../outputs/throughput_results/"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Normalize fields
    df['BSS Id'] = df['BSS Id'].astype(str).str.lower()
    df['Receiver address'] = df['Receiver address'].astype(str).str.lower()
    df['Type/Subtype'] = df['Type/Subtype'].astype(str).str.lower()
    ap_bssid = ap_bssid.lower()

    # Filter downlink Data/QoS frames
    downlink = df[
        (df['BSS Id'] == ap_bssid) &
        (df['Type/Subtype'].str.contains("data"))
    ].copy()

    downlink['Timestamp'] = pd.to_numeric(downlink['TSF Timestamp'], errors='coerce')
    downlink = downlink.dropna(subset=['Timestamp'])

    if downlink.empty:
        print("No data frames found from the AP.")
        return

    clients = downlink['Receiver address'].unique()
    clients = [mac for mac in clients if is_unicast_mac(mac)]

    if not clients:
        print("No unicast client devices found.")
        return

    sns.set_theme(style="whitegrid")
    summary = []

    for device in clients:
        client = downlink[downlink['Receiver address'] == device].copy()
        if client.empty:
            continue

        client['Time (s)'] = (client['Timestamp'] - client['Timestamp'].min()) / 1e6
        client['Second'] = client['Time (s)'].astype(int)

        assumed_bits_per_frame = 12000
        throughput = (
            client.groupby('Second').size() * assumed_bits_per_frame / 1e6
        ).reset_index(name='Throughput (Mbps)')

        retry_count = client[client['Retry'].astype(str) == "1"].shape[0]
        total_frames = client.shape[0]
        frame_loss_rate = retry_count / total_frames if total_frames > 0 else 0

        data_rates = client['Data rate'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
        data_rates = pd.to_numeric(data_rates, errors='coerce').dropna()
        avg_data_rate = data_rates.mean() if not data_rates.empty else 0
        estimated_throughput = avg_data_rate * (1 - frame_loss_rate)

        mcs = pd.to_numeric(client['MCS Index'], errors='coerce').mode()
        mcs = int(mcs.iloc[0]) if not mcs.empty else -1
        phy = client['PHY type'].mode().iloc[0] if 'PHY type' in client and not client['PHY type'].mode().empty else "N/A"
        bw = client['Bandwidth'].mode().iloc[0] if 'Bandwidth' in client and not client['Bandwidth'].mode().empty else "N/A"
        sgi = client['Short GI'].mode().iloc[0] if 'Short GI' in client and not client['Short GI'].mode().empty else "N/A"

        signal = client['Signal strength'].astype(str).str.extract(r'(-?\d+)')[0]
        signal = pd.to_numeric(signal, errors='coerce').dropna()
        avg_signal = signal.mean() if not signal.empty else "N/A"

        avg_obs_tp = throughput['Throughput (Mbps)'].mean()
        rate_gap = avg_data_rate - avg_obs_tp

        percentiles = throughput['Throughput (Mbps)'].quantile([0.25, 0.5, 0.75, 0.95])
        tp_min = throughput['Throughput (Mbps)'].min()
        tp_max = throughput['Throughput (Mbps)'].max()
        tp_25p = percentiles[0.25]
        tp_50p = percentiles[0.5]
        tp_75p = percentiles[0.75]
        tp_95p = percentiles[0.95]

        # Diagnosis
        if avg_signal != "N/A" and avg_signal < -70:
            reason = "Weak signal"
        elif frame_loss_rate > 0.1:
            reason = "High frame loss"
        elif avg_data_rate < 24:
            reason = "Low PHY rate"
        elif rate_gap > 50:
            reason = "Rate gap → possible interference"
        else:
            reason = "No obvious issues"

        # Time series plot
        throughput['3-point MA'] = throughput['Throughput (Mbps)'].rolling(3, min_periods=1).mean()
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=throughput, x='Second', y='Throughput (Mbps)', label='Actual')
        sns.lineplot(data=throughput, x='Second', y='3-point MA', label='3-pt Moving Avg', linestyle='--')
        plt.title(f"Throughput Over Time - {device}")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (Mbps)")
        plt.legend()
        plt.tight_layout()
        plot_name = f"throughput_{device.replace(':', '')}.png"
        plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()

        print(f"\nDevice: {device}")
        print(f"   → Frames: {total_frames}")
        print(f"   → Retries: {retry_count}")
        print(f"   → Avg Throughput: {avg_obs_tp:.3f} Mbps")
        print(f"   → Rate Gap: {rate_gap:.2f} Mbps")
        print(f"   → Diagnosis: {reason}")

        summary.append({
            'Client': device,
            'Frames': total_frames,
            'Retries': retry_count,
            'Retry Rate (%)': round(frame_loss_rate * 100, 2),
            'Avg Data Rate (Mbps)': round(avg_data_rate, 2),
            'Estimated Throughput (Mbps)': round(estimated_throughput, 3),
            'Observed Throughput (Mbps)': round(avg_obs_tp, 3),
            'Min TP (Mbps)': round(tp_min, 3),
            '25P TP (Mbps)': round(tp_25p, 3),
            'Median TP (Mbps)': round(tp_50p, 3),
            '75P TP (Mbps)': round(tp_75p, 3),
            '95P TP (Mbps)': round(tp_95p, 3),
            'Max TP (Mbps)': round(tp_max, 3),
            'Rate Gap (Mbps)': round(rate_gap, 3),
            'Signal Strength (dBm)': round(avg_signal, 1) if avg_signal != "N/A" else "N/A",
            'MCS Index': mcs,
            'PHY Type': phy,
            'Bandwidth': bw,
            'Short GI': sgi,
            'Diagnosis': reason
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "throughput_summary.csv"), index=False)

    # Plot: Histogram of Observed Throughput binned by Signal Strength
    valid_corr = summary_df[summary_df['Signal Strength (dBm)'] != "N/A"].copy()
    valid_corr['Signal Strength (dBm)'] = pd.to_numeric(valid_corr['Signal Strength (dBm)'], errors='coerce')
    valid_corr = valid_corr.dropna(subset=['Signal Strength (dBm)', 'Observed Throughput (Mbps)'])

    if not valid_corr.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=valid_corr,
            x='Signal Strength (dBm)',
            weights='Observed Throughput (Mbps)',
            bins=30,
            hue='Diagnosis',
            multiple='stack',
            color='steelblue',
        )
        plt.title("Observed Throughput Distribution by Signal Strength (Binned)")
        plt.ylabel("Total Throughput (Mbps)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "binned_throughput_by_signal.png"))
        plt.close()

    # Plot 2: Rate Gap per Client
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x='Client', y='Rate Gap (Mbps)')
    plt.title("Rate Gap per Client")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rate_gap_per_client.png"))
    plt.close()

    # Plot 3: Throughput Percentiles per Client (Expanded with 25P)
    melted = summary_df.melt(
        id_vars='Client',
        value_vars=['Min TP (Mbps)', '25P TP (Mbps)', 'Median TP (Mbps)', '75P TP (Mbps)', '95P TP (Mbps)'],
        var_name='Metric', value_name='Mbps'
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Client', y='Mbps', hue='Metric')
    plt.title("Throughput Distribution per Client (Min / 25P / Median / 75P / 95P)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_distribution.png"))
    plt.close()

    print(f"\nCompleted: {len(summary_df)} devices analyzed.")
    print(f"All plots and CSV saved to: {output_dir}")


if __name__ == "__main__":
    csv_path = "../results_parser/home_parsed_packets.csv"
    ap_bssid = "0c:67:14:92:8c:44"
    analyze_ap_clients(csv_path, ap_bssid)
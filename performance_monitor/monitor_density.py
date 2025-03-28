import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_wifi_density(csv_path, output_dir="../outputs/density_results/"):
    os.makedirs(output_dir, exist_ok=True)

    # Load and clean data
    df = pd.read_csv(csv_path)

    # Filter only beacon frames
    beacons = df[df['Type/Subtype'].str.contains('Beacon', case=False)].copy()
    if beacons.empty:
        print("No Beacon frames found.")
        return

    # Normalize values
    beacons['Frequency (MHz)'] = pd.to_numeric(
        beacons['Frequency'].astype(str).str.extract(r'(\d+)')[0],
        errors='coerce'
    )
    beacons['Signal strength (dBm)'] = pd.to_numeric(
        beacons['Signal strength'].str.extract(r'(-?\d+)')[0],
        errors='coerce'
    )
    beacons['Signal/Noise Ratio (dB)'] = pd.to_numeric(
        beacons['Signal/noise ratio'].astype(str).str.extract(r'(\d+)')[0],
        errors='coerce'
    )
    beacons.dropna(subset=['Frequency (MHz)'], inplace=True)

    # ========== Summary Table ==========
    summary = (
        beacons.groupby('Frequency (MHz)').agg({
            'BSS Id': pd.Series.nunique,
            'Signal strength (dBm)': 'mean',
            'Signal/Noise Ratio (dB)': 'mean'
        }).reset_index()
        .rename(columns={
            'BSS Id': 'Unique BSSIDs',
            'Signal strength (dBm)': 'Avg Signal Strength (dBm)',
            'Signal/Noise Ratio (dB)': 'Avg SNR (dB)'
        })
        .sort_values(by='Frequency (MHz)')
    )

    print("\nWi-Fi Density Summary:\n")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(output_dir, "density_summary.csv"), index=False)

    # ========== Base Styling ==========
    sns.set_theme(style="whitegrid")
    blue = "steelblue"

    # SNR per BSSID Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=beacons, x='BSS Id', y='Signal/Noise Ratio (dB)', color="#4682B4")
    plt.xticks(rotation=90)
    plt.title("SNR Distribution per BSSID")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "snr_per_bssid.png"))
    plt.close()

    # === Clean Signal/noise ratio globally ===
    if df['Signal/noise ratio'].dtype != 'float':
        df['Signal/noise ratio'] = df['Signal/noise ratio'].astype(str).str.replace(" dB", "", regex=False)
        df['Signal/noise ratio'] = pd.to_numeric(df['Signal/noise ratio'], errors='coerce')

    df = df.dropna(subset=['Signal/noise ratio'])

    # Classify SNR zones
    def classify_snr(snr):
        if snr > 25:
            return 'Good'
        elif snr > 10:
            return 'Moderate'
        else:
            return 'Poor'

    df['SNR Quality'] = df['Signal/noise ratio'].apply(classify_snr)

    # Barplot for SNR Quality
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='SNR Quality', hue='SNR Quality', order=['Poor', 'Moderate', 'Good'], palette='Blues')
    plt.title("SNR Quality Classification", fontsize=14)
    plt.xlabel("SNR Category", fontsize=12)
    plt.ylabel("Packet Count", fontsize=12)
    plt.legend(title="SNR Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "snr_quality_barplot.png"))
    plt.close()

    # Histogram: Signal Strength
    plt.figure(figsize=(10, 6))
    sns.histplot(data=beacons, x='Signal strength (dBm)', bins=30, kde=True, color=blue)
    plt.title("Signal Strength Distribution (All Frequencies)")
    plt.xlabel("Signal Strength (dBm)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_strength_distribution.png"))
    plt.close()

    # Violin plot for Signal Strength by Frequency
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=beacons, x='Frequency (MHz)', y='Signal strength (dBm)', inner="quartile", color=blue)
    plt.title("Signal Strength Distribution by Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "violinplot_signal_strength.png"))
    plt.close()

    # Histogram: SNR Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Signal/noise ratio', bins=30, kde=True, color=blue)
    plt.title("SNR Distribution Across All Packets", fontsize=14)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Packet Count")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "snr_distribution_plot.png"))
    plt.close()

    # Channel Heatmap
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    heatmap_df = df[df['Type/Subtype'].str.contains('Beacon', case=False)].dropna(subset=['Frequency'])

    heatmap_data = heatmap_df.groupby(['Frequency', 'BSS Id']).size().reset_index(name='Count')
    heatmap_pivot = heatmap_data.pivot_table(index='BSS Id', columns='Frequency', values='Count', fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, cmap='Blues', linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Beacon Count'})
    plt.title('Wi-Fi Channel Heatmap (Beacon Frames per BSSID)', fontsize=15)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('BSSID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_heatmap.png'))
    plt.close()

    print(f"\nAll density plots and summaries saved to: {output_dir}")
    return summary


# Optional direct run
if __name__ == "__main__":
    csv_path = "../results_parser/home_Network5G_parsed_packets.csv"
    analyze_wifi_density(csv_path)

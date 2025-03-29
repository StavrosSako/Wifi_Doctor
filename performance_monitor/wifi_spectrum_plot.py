import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_spectrum_coverage(csv_path, output_dir="../outputs/density_results/"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    df['Signal strength (dBm)'] = pd.to_numeric(df['Signal strength'].str.extract(r'(-?\d+)')[0], errors='coerce')
    df['Bandwidth'] = df['Bandwidth'].str.extract(r'(\d+)').astype(float)

    # Use only valid beacons with frequency and BSSID
    beacons = df[df['Type/Subtype'].str.contains("Beacon", case=False)].dropna(subset=['Frequency', 'BSS Id'])

    # Get strongest signal for each BSSID + Frequency
    spectrum = (
        beacons.groupby(['BSS Id', 'Frequency', 'Bandwidth'])['Signal strength (dBm)']
        .min()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, row in spectrum.iterrows():
        freq = row['Frequency']
        bw = row['Bandwidth'] if not pd.isna(row['Bandwidth']) else 20
        bssid = row['BSS Id']
        strength = row['Signal strength (dBm)']

        # Rectangle range: full bandwidth span centered on frequency
        left = freq - (bw / 2)
        rect = patches.Rectangle(
            (left, strength),
            bw,
            2,  # height of bar
            alpha=0.5,
            label=bssid if i == 0 or bssid not in spectrum['BSS Id'][:i].values else "",
        )
        ax.add_patch(rect)
        ax.text(freq, strength + 1, bssid[-5:], ha='center', fontsize=8, alpha=0.8)

    ax.set_xlim(4900, 5900)
    ax.set_ylim(-100, -30)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Signal Strength (dBm)")
    ax.set_title("Wi-Fi Channel Spectrum (with Bandwidth Overlay)")
    ax.grid(True)
    ax.legend(title="BSSIDs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_file = os.path.join(output_dir, "spectrum_band_chart.png")
    plt.savefig(output_file)
    plt.close()
    print(f"[✓] Spectrum coverage chart saved to {output_file}")

# Run
if __name__ == "__main__":
    csv_path = "../results_parser/home_Network_5G_parsed_packets.csv"
    generate_spectrum_coverage(csv_path)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load parsed packet data
csv_path = '../results_parser/HowIWiFi_PCAP_parsed_packets.csv'
df = pd.read_csv(csv_path)

# Clean SNR column and convert to numeric
df['Signal/noise ratio'] = df['Signal/noise ratio'].str.replace(" dB", "", regex=False)
df['Signal/noise ratio'] = pd.to_numeric(df['Signal/noise ratio'], errors='coerce')
df = df.dropna(subset=['Signal/noise ratio'])

# Define SNR quality zones
def classify_snr(snr):
    if snr > 25:
        return 'Good'
    elif snr > 10:
        return 'Moderate'
    else:
        return 'Poor'

df['SNR Quality'] = df['Signal/noise ratio'].apply(classify_snr)

# Plot histogram with KDE
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
sns.histplot(data=df, x='Signal/noise ratio', bins=30, kde=True, color='steelblue')

# Styling
plt.title("Wi-Fi Signal-to-Noise Ratio (SNR) Distribution", fontsize=15, weight='bold')
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Packet Count", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="SNR Quality")

# Save figure
output_path = '../monitor_results/snr_distribution_plot.png'
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
plt.close()

output_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load CSV
input_file = '../results_parser/HowIWiFi_PCAP_parsed_packets.csv'
df = pd.read_csv(input_file)

# Clean numeric values
df['Data rate'] = pd.to_numeric(df['Data rate'].str.replace(' Mbps', ''), errors='coerce')
df['Signal/noise ratio'] = pd.to_numeric(df['Signal/noise ratio'].str.replace(' dB', ''), errors='coerce')
df['PHY type'] = df['PHY type'].astype(str)

# Filter necessary rows
df_filtered = df.dropna(subset=['Data rate', 'Signal/noise ratio'])

# Assume bandwidth based on PHY
phy_bandwidth_map = {
    'PHY type: 802.11g (ERP) (6)': 20,
    'PHY type: 802.11n (HT) (7)': 20,
    'PHY type: 802.11b (HR/DSSS) (4)': 22
}
df_filtered['Assumed Bandwidth'] = df_filtered['PHY type'].map(phy_bandwidth_map)
df_filtered = df_filtered.dropna(subset=['Assumed Bandwidth'])

# Calculate expected rate and rate gap
df_filtered['Expected Rate'] = df_filtered['Signal/noise ratio'] * df_filtered['Assumed Bandwidth'] * 0.1
df_filtered['Rate Gap'] = df_filtered['Expected Rate'] - df_filtered['Data rate']

# Save CSV results
output_csv = "../monitor_results/rate_gap_results.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_filtered.to_csv(output_csv, index=False)
print(f" Rate gap data saved to: {output_csv}")

# Plot distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered['Rate Gap'], bins=20, kde=True, color='steelblue')
plt.title("Rate Gap Distribution (Expected Rate - Actual Rate)", fontsize=14)
plt.xlabel("Rate Gap (Mbps)", fontsize=12)
plt.ylabel("Packet Count", fontsize=12)
plt.grid(True)

# Save the plot
output_plot = "../monitor_results/rate_gap_distribution.png"
plt.savefig(output_plot, bbox_inches='tight')
plt.close()
print(f"Rate gap plot saved to: {output_plot}")

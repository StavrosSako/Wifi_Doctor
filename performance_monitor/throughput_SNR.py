import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load parsed packet data from CSV
df = pd.read_csv('../results_parser/home_Network5G_parsed_packets.csv')

# Clean and convert SNR and Timestamp
df = df[df['Signal/noise ratio'] != 'N/A']
df['SNR (dB)'] = df['Signal/noise ratio'].str.extract(r'(-?\d+)').astype(float)

# Convert TSF Timestamp to numeric
df = df[df['TSF Timestamp'] != 'Absent']
df['Timestamp'] = pd.to_numeric(df['TSF Timestamp'], errors='coerce')

# Drop rows with missing timestamps
df = df.dropna(subset=['Timestamp'])

# Sort by timestamp and calculate throughput per interval (e.g., 1 second)
df = df.sort_values('Timestamp')

# Normalize timestamps to seconds (TSF is in microseconds)
df['Timestamp'] = df['Timestamp'] / 1e6

# Set interval (1s buckets)
df['TimeBucket'] = (df['Timestamp']).astype(int)

# Assume each packet carries 1500 bytes
df['Throughput (Mbps)'] = 1500 * 8 / 1e6  # Megabits per packet

# Group by time bucket and average SNR + sum throughput
snr_throughput = df.groupby('TimeBucket').agg({
    'SNR (dB)': 'mean',
    'Throughput (Mbps)': 'sum'
}).reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=snr_throughput, x='SNR (dB)', y='Throughput (Mbps)', color='steelblue')
sns.regplot(data=snr_throughput, x='SNR (dB)', y='Throughput (Mbps)', scatter=False, color='darkblue', line_kws={"linewidth": 2})

plt.title("Relationship Between SNR and Throughput", fontsize=14)
plt.xlabel("Average SNR (dB)", fontsize=12)
plt.ylabel("Throughput (Mbps per second)", fontsize=12)
plt.grid(True)

# Save the plot
output_path = "../monitor_results/snr_vs_throughput1.png"
plt.savefig(output_path, bbox_inches='tight')
plt.close()

output_path

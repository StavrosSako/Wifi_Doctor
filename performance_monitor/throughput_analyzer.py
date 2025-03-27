import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Input and Output Paths
input_file = '../results_parser/HowIWiFi_PCAP_parsed_packets.csv'
output_csv = '../monitor_results/throughput_summary.csv'
output_plot = '../monitor_results/throughput_by_phy.png'

# Load data
df = pd.read_csv(input_file)

# Clean fields
df['PHY type'] = df['PHY type'].str.extract(r'PHY type:\s*([^\(]+)', expand=False).str.strip()
df['Data rate'] = pd.to_numeric(df['Data rate'].str.replace(" Mbps", ""), errors='coerce')

# Assume 5% retransmission = 5% frame loss (you can later refine this)
frame_loss_rate = 0.05
df['Estimated Throughput (Mbps)'] = df['Data rate'] * (1 - frame_loss_rate)

# Drop rows without PHY or Data rate
df_clean = df.dropna(subset=['PHY type', 'Estimated Throughput (Mbps)'])

# Group by PHY type and compute average throughput
summary = df_clean.groupby('PHY type').agg({
    'Estimated Throughput (Mbps)': 'mean',
    'PHY type': 'count'
}).rename(columns={
    'PHY type': 'Packet Count',
    'Estimated Throughput (Mbps)': 'Avg Throughput (Mbps)'
}).reset_index()

# Save summary as CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
summary.to_csv(output_csv, index=False)
print(f"Throughput summary saved to: {output_csv}")

# Plot Average Throughput per PHY Type
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
steel_blue = sns.color_palette("Blues", n_colors=4)[2:]

ax = sns.barplot(x='PHY type', y='Avg Throughput (Mbps)', data=summary, palette=steel_blue)

# Add labels
for index, row in summary.iterrows():
    ax.text(index, row['Avg Throughput (Mbps)'] + 0.5, f"{row['Avg Throughput (Mbps)']:.1f}", 
            ha='center', fontsize=10)

plt.title("Estimated Avg Throughput by PHY Type", fontsize=14, weight='bold')
plt.xlabel("PHY Type", fontsize=12)
plt.ylabel("Throughput (Mbps)", fontsize=12)
plt.tight_layout()

# Save plot
plt.savefig(output_plot, bbox_inches='tight')
plt.close()
print(f"Throughput plot saved to: {output_plot}")

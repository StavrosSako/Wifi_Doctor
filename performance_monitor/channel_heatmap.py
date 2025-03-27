import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load parsed packet data from CSV
input_file = '../results_parser/HowIWiFi_PCAP_parsed_packets.csv'
df = pd.read_csv(input_file)

# Clean and convert channel data
df['Channel'] = pd.to_numeric(df['Channel'], errors='coerce')
df = df.dropna(subset=['Channel'])
df['Channel'] = df['Channel'].astype(int)

# Count packets per channel
channel_counts = df['Channel'].value_counts().sort_index()

# Match color style from PHY/Data Rate plots
color_palette = sns.color_palette("Blues", n_colors=len(channel_counts))

# Plot professionally styled heatmap bar chart
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

bars = sns.barplot(
    x=channel_counts.index,
    y=channel_counts.values,
    palette=color_palette,
    hue=channel_counts.index,
    legend=False
)

# Add value labels above bars
for bar in bars.patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=9,
        color='black'
    )

# Styling
plt.title("Wi-Fi Channel Utilization Heatmap (Packet Count per Channel)", fontsize=15, weight='bold')
plt.xlabel("Wi-Fi Channel", fontsize=12)
plt.ylabel("Packet Count", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Save figure
output_path = "./monitor_results/channel_heatmap.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
plt.close()

output_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load parsed CSV
csv_path = '../results_parser/HowIWiFi_PCAP_parsed_packets.csv'
df = pd.read_csv(csv_path)

# Clean data
df['PHY type'] = df['PHY type'].fillna('N/A')
df['Data rate'] = df['Data rate'].str.replace(" Mbps", "", regex=False)
df['Data rate'] = pd.to_numeric(df['Data rate'], errors='coerce')

# Simplify PHY labels
df['PHY type'] = df['PHY type'].str.extract(r'(802\.11[a-z]+)', expand=False).fillna('Unknown')

# Count for PHY type
phy_counts = df['PHY type'].value_counts()

# Set professional style
sns.set_style("white")
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
sns.barplot(x=phy_counts.index, y=phy_counts.values, palette='Blues_d')
plt.title("Wi-Fi PHY Type Distribution", fontsize=16, weight='bold')
plt.ylabel("Number of Packets", fontsize=12)
plt.xlabel("PHY Type", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for i, val in enumerate(phy_counts.values):
    plt.text(i, val + 1, str(val), ha='center', fontsize=9, color='black')
    
plt.subplot(2, 1, 2)
sns.histplot(df['Data rate'].dropna(), bins=30, kde=True, color="steelblue")
plt.title("Wi-Fi Data Rate Distribution", fontsize=16, weight='bold')
plt.xlabel("Data Rate (Mbps)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Save to file
plot_path = '../monitor_results/phy_datarate_plot.png'
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

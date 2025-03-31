import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from collections import defaultdict
import matplotlib.gridspec as gridspec

# Set style for professional plots
plt.style.use('seaborn-v0_8')
# Set custom color palette with professional blues
custom_palette = ['#1f77b4', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7']
sns.set_palette(custom_palette)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def create_visualization_directory():
    """Create directory for saving visualizations if it doesn't exist."""
    # Get the absolute path of the script's directory
    script_dir = Path(__file__).resolve().parent
    # Get the project root directory (one level up from script_dir)
    project_root = script_dir.parent
    # Create the visualization_results directory in project root
    parent_dir = project_root / 'visualization_results'
    parent_dir.mkdir(exist_ok=True)
    
    # Create both throughput and density directories
    throughput_dir = parent_dir / 'throughput'
    density_dir = parent_dir / 'density'
    throughput_dir.mkdir(exist_ok=True)
    density_dir.mkdir(exist_ok=True)
    
    return throughput_dir, density_dir

def map_phy_type(phy_value):
    """Map PHY type numbers to their descriptions."""
    phy_map = {
        1: '802.11b',
        2: '802.11b',
        3: '802.11g',
        4: '802.11g',
        5: '802.11a',
        6: '802.11a',
        7: '802.11n',
        8: '802.11n',
        9: '802.11n',
        10: '802.11n',
        11: '802.11ac',
        12: '802.11ac',
        13: '802.11ac',
        14: '802.11ac',
        15: '802.11ax',
        16: '802.11ax',
        17: '802.11ax',
        18: '802.11ax'
    }
    return phy_map.get(int(phy_value), f'Unknown ({phy_value})')

def categorize_channel(channel):
    """Categorize channel into 2.4GHz or 5GHz band."""
    try:
        channel = int(channel)
        if 1 <= channel <= 14:
            return '2.4 GHz'
        elif channel >= 36:  # 5GHz channels start from 36
            return '5 GHz'
        else:
            return 'Unknown'
    except:
        return 'Unknown'

def plot_ap_count_per_channel(df, output_dir):
    """Plot the count of unique access points per channel."""
    plt.figure(figsize=(12, 6))
    
    # Count unique BSSIDs per channel
    channel_counts = df.groupby('Actual Channel')['BSS Id'].nunique()
    
    # Create bar plot
    bars = plt.bar(channel_counts.index, channel_counts.values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title('Unique Access Points per Channel')
    plt.xlabel('Channel Number')
    plt.ylabel('Number of Unique Access Points')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_dir / 'ap_count_per_channel.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_signal_strength_distribution(df, output_dir):
    """Plot the distribution of signal strength with KDE."""
    plt.figure(figsize=(12, 6))
    
    # Create histogram with KDE
    sns.histplot(data=df, x='Signal strength', bins=30, kde=True)
    
    # Add vertical lines for mean and median
    plt.axvline(df['Signal strength'].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(df['Signal strength'].median(), color='green', linestyle='--', label='Median')
    
    # Add statistics box
    stats_text = f"Mean: {df['Signal strength'].mean():.1f} dBm\n"
    stats_text += f"Median: {df['Signal strength'].median():.1f} dBm\n"
    stats_text += f"Std Dev: {df['Signal strength'].std():.1f} dBm"
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Signal Strength Distribution')
    plt.xlabel('Signal Strength (dBm)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_dir / 'signal_strength_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_snr_distribution(df, output_dir):
    """Plot the distribution of Signal-to-Noise Ratio with KDE."""
    plt.figure(figsize=(12, 6))
    
    # Create histogram with KDE
    sns.histplot(data=df, x='Signal/noise ratio', bins=30, kde=True)
    
    # Add vertical lines for mean and median
    plt.axvline(df['Signal/noise ratio'].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(df['Signal/noise ratio'].median(), color='green', linestyle='--', label='Median')
    
    # Add statistics box
    stats_text = f"Mean: {df['Signal/noise ratio'].mean():.1f} dB\n"
    stats_text += f"Median: {df['Signal/noise ratio'].median():.1f} dB\n"
    stats_text += f"Std Dev: {df['Signal/noise ratio'].std():.1f} dB"
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Signal-to-Noise Ratio Distribution')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_dir / 'snr_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()




def generate_wifi_density_summary(df, output_dir):
    """Generate a summary table of WiFi density statistics."""
    # Map PHY types
    df['PHY Type'] = df['PHY type'].apply(map_phy_type)
    
    # Count unique BSSIDs per channel
    channel_counts = df.groupby('Actual Channel')['BSS Id'].nunique()
    
    # Count unique BSSIDs per PHY type
    phy_counts = df.groupby('PHY Type')['BSS Id'].nunique()
    
    summary = {
        'Total APs': len(df['BSS Id'].unique()),
        'Total Channels': len(df['Actual Channel'].unique()),
        'Average RSSI': df['Signal strength'].mean(),
        'Average SNR': df['Signal/noise ratio'].mean(),
        'Most Common Channel': channel_counts.idxmax(),
        'Most Common PHY Type': phy_counts.idxmax(),
        'Channel Distribution': channel_counts.to_dict(),
        'PHY Type Distribution': phy_counts.to_dict()
    }
    
    # Save summary to CSV
    pd.DataFrame([summary]).to_csv(output_dir / 'wifi_density_summary.csv', index=False)
    
    # Save summary to text file
    with open(output_dir / 'wifi_density_summary.txt', 'w') as f:
        f.write("WiFi Network Analysis Summary\n")
        f.write("==========================\n\n")
        f.write(f"Total Access Points: {summary['Total APs']}\n")
        f.write(f"Total Channels Used: {summary['Total Channels']}\n")
        f.write(f"Average Signal Strength (RSSI): {summary['Average RSSI']:.1f} dBm\n")
        f.write(f"Average Signal-to-Noise Ratio (SNR): {summary['Average SNR']:.1f} dB\n")
        f.write(f"Most Common Channel: {summary['Most Common Channel']}\n")
        f.write(f"Most Common PHY Type: {summary['Most Common PHY Type']}\n\n")
        
        f.write("Channel Distribution (Unique APs):\n")
        for channel, count in summary['Channel Distribution'].items():
            f.write(f"Channel {channel}: {count} APs\n")
        
        f.write("\nPHY Type Distribution (Unique APs):\n")
        for phy_type, count in summary['PHY Type Distribution'].items():
            f.write(f"{phy_type}: {count} APs\n")


def plot_ap_stability(df, output_dir):
    """Analyze and plot AP signal stability over time with band distinction."""
    if 'Band' not in df.columns:
        df['Band'] = df['Actual Channel'].apply(categorize_channel)
    
    plt.figure(figsize=(12, 6))
    
    # Group by BSS ID and calculate signal strength statistics
    ap_stats = df.groupby(['BSS Id', 'Band'])['Signal strength'].agg(['mean', 'std', 'count']).reset_index()
    ap_stats = ap_stats[ap_stats['count'] > 10]  # Filter APs with enough samples
    
    # Create scatter plot with different colors for each band
    for band in ap_stats['Band'].unique():
        band_data = ap_stats[ap_stats['Band'] == band]
        plt.scatter(band_data['mean'], band_data['std'], 
                   alpha=0.6, 
                   label=band,
                   marker='o' if band == '2.4 GHz' else '^')
    
    plt.xlabel('Mean Signal Strength (dBm)')
    plt.ylabel('Signal Strength Standard Deviation')
    plt.title('AP Signal Stability Analysis by Band')
    plt.legend()
    
    # Add annotations for unstable APs
    unstable_aps = ap_stats[ap_stats['std'] > ap_stats['std'].mean()]
    for _, ap in unstable_aps.iterrows():
        plt.annotate(f"{ap['BSS Id'][-8:]} ({ap['Band']})", 
                    (ap['mean'], ap['std']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'ap_stability.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_network_coverage(df, output_dir):
    """Analyze and plot network coverage quality."""
    plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2)
    
    # Signal Strength Distribution by AP
    ax1 = plt.subplot(gs[0, 0])
    # Use violin plot instead of boxplot for better distribution visualization
    sns.violinplot(data=df, x='BSS Id', y='Signal strength', ax=ax1, inner='box')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Signal Strength Distribution by AP')
    
    # Coverage Quality Categories
    ax2 = plt.subplot(gs[0, 1])
    df['Coverage Quality'] = pd.cut(df['Signal strength'],
                                  bins=[-100, -80, -70, -60, -50, 0],
                                  labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    coverage_counts = df['Coverage Quality'].value_counts()
    coverage_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_title('Coverage Quality Distribution')
    
    # Signal to Noise Ratio Distribution
    ax3 = plt.subplot(gs[1, :])
    sns.histplot(data=df, x='Signal/noise ratio', bins=30, ax=ax3)
    ax3.set_title('Signal to Noise Ratio Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'network_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_channel_utilization_patterns(df, output_dir):
    """Analyze and plot channel utilization patterns."""
    plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2)
    
    # Channel Usage Over Time
    ax1 = plt.subplot(gs[0, 0])
    channel_counts = df['Actual Channel'].value_counts()
    channel_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Channel Usage Distribution')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Number of Frames')
    
    # Channel vs Bandwidth
    ax2 = plt.subplot(gs[0, 1])
    sns.scatterplot(data=df, x='Actual Channel', y='Bandwidth', ax=ax2)
    ax2.set_title('Channel vs Bandwidth')
    
    # Channel Interference Analysis
    ax3 = plt.subplot(gs[1, :])
    channel_snr = df.groupby('Actual Channel')['Signal/noise ratio'].agg(['mean', 'std']).reset_index()
    ax3.errorbar(channel_snr['Actual Channel'], channel_snr['mean'], 
                yerr=channel_snr['std'], fmt='o')
    ax3.set_title('Channel Quality (SNR) Analysis')
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Signal to Noise Ratio (mean ± std)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_network_quality_metrics(df, output_dir):
    """Analyze and plot comprehensive network quality metrics."""
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # Signal Quality Over Time
    ax1 = plt.subplot(gs[0, 0])
    df['Signal Quality'] = (df['Signal strength'] - df['Signal strength'].min()) / \
                          (df['Signal strength'].max() - df['Signal strength'].min()) * 100
    df['Sample Index'] = range(len(df))  # Add index for x-axis
    
    # Calculate moving average for smoother trend
    window_size = 10
    df['Signal Quality MA'] = df['Signal Quality'].rolling(window=window_size, center=True).mean()
    
    # Plot both raw data and moving average
    sns.lineplot(data=df, x='Sample Index', y='Signal Quality', alpha=0.3, label='Raw Data', ax=ax1)
    sns.lineplot(data=df, x='Sample Index', y='Signal Quality MA', color='red', label=f'{window_size}-sample Moving Average', ax=ax1)
    
    # Set y-axis limits to focus on the main range of values
    y_min = df['Signal Quality'].min()
    y_max = df['Signal Quality'].max()
    y_range = y_max - y_min
    ax1.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
    
    ax1.set_title('Signal Quality Trend')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Signal Quality (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SNR Quality Categories
    ax2 = plt.subplot(gs[0, 1])
    df['SNR Quality'] = pd.cut(df['Signal/noise ratio'],
                              bins=[-np.inf, 10, 20, 30, 40, np.inf],
                              labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
    snr_counts = df['SNR Quality'].value_counts()
    snr_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_title('SNR Quality Distribution')
    
    # Network Load
    ax3 = plt.subplot(gs[1, :])
    ap_load = df.groupby('BSS Id').size()
    ap_load.plot(kind='bar', ax=ax3)
    ax3.set_title('Network Load by AP')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'network_quality.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary(df, output_dir):
    """Generate an enhanced summary of the WiFi environment with band distinction."""
    if 'Band' not in df.columns:
        df['Band'] = df['Actual Channel'].apply(categorize_channel)
    
    # Separate statistics by band
    band_stats = {}
    for band in df['Band'].unique():
        band_data = df[df['Band'] == band]
        band_stats[band] = {
            'Total APs': len(band_data['BSS Id'].unique()),
            'Total Channels': len(band_data['Actual Channel'].unique()),
            'Average Signal Strength': f"{band_data['Signal strength'].mean():.2f} dBm",
            'Average SNR': f"{band_data['Signal/noise ratio'].mean():.2f}",
            'Most Used Channel': band_data['Actual Channel'].mode().iloc[0],
            'Average Data Rate': f"{band_data['Data Rate'].mean():.2f} Mbps",
            'Signal Quality Stats': {
                'Excellent (>-50dBm)': len(band_data[band_data['Signal strength'] > -50]),
                'Good (-50 to -60dBm)': len(band_data[(band_data['Signal strength'] <= -50) & 
                                                     (band_data['Signal strength'] > -60)]),
                'Fair (-60 to -70dBm)': len(band_data[(band_data['Signal strength'] <= -60) & 
                                                     (band_data['Signal strength'] > -70)]),
                'Poor (<-70dBm)': len(band_data[band_data['Signal strength'] <= -70])
            }
        }
    
    # Save as text file with band distinction
    with open(output_dir / 'wifi_summary.txt', 'w') as f:
        f.write("WiFi Environment Summary by Band\n")
        f.write("==============================\n\n")
        
        for band, stats in band_stats.items():
            f.write(f"\n{band} Band Statistics:\n")
            f.write("=" * (len(band) + 15) + "\n")
            for key, value in stats.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
    
    # Save as CSV with band distinction
    flat_summary = {}
    for band, stats in band_stats.items():
        band_prefix = band.replace(" ", "_")
        for key, value in stats.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_summary[f"{band_prefix}_{key}_{subkey}"] = subvalue
            else:
                flat_summary[f"{band_prefix}_{key}"] = value
    
    pd.DataFrame([flat_summary]).to_csv(output_dir / 'wifi_summary.csv', index=False)

def plot_time_based_analysis(df, output_dir):
    """Analyze and plot time-based patterns in the network."""
    plt.figure(figsize=(12, 6))
    
    # Signal Strength Over Time
    df['Sample Index'] = range(len(df))  # Use row index as time proxy
    for bss_id in df['BSS Id'].unique()[:8]:  # Plot top 8 APs
        ap_data = df[df['BSS Id'] == bss_id]
        plt.plot(ap_data['Sample Index'], ap_data['Signal strength'], label=bss_id, alpha=0.7)
    
    plt.title('Signal Strength Over Time (Top 8 APs)')
    plt.xlabel('Sample Number')
    plt.ylabel('Signal Strength (dBm)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_based_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_wifi_density(csv_file):
    """Main function to analyze WiFi network data and generate visualizations."""
    # Create output directories
    throughput_dir, density_dir = create_visualization_directory()
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Generate all visualizations
    plot_ap_count_per_channel(df, density_dir)
    plot_signal_strength_distribution(df, density_dir)
    plot_snr_distribution(df, density_dir)
    generate_wifi_density_summary(df, density_dir)
    plot_ap_stability(df, density_dir)
    plot_network_coverage(df, density_dir)
    plot_network_quality_metrics(df, density_dir)
    plot_time_based_analysis(df, density_dir)
    generate_summary(df, density_dir)
    
    print(f"Enhanced analysis complete. Results saved in {density_dir}")

if __name__ == "__main__":
    # Get the absolute path of the script's directory
    script_dir = Path(__file__).resolve().parent
    # Get the project root directory (one level up from script_dir)
    project_root = script_dir.parent
    # Use the correct path to the CSV file
    csv_file = project_root / 'results_parser' / 'density_analysis.csv'
    analyze_wifi_density(csv_file)

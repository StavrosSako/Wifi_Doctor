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

# Set global plotting parameters
plt.rcParams.update({
    'figure.figsize': (16, 9),  # Standard 16:9 aspect ratio
    'figure.dpi': 100,
    'savefig.dpi': 200,  # Reduced DPI for output
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.autolayout': True  # Use automatic layout instead of constrained
})

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

def calculate_throughput(df):
    """Calculate theoretical throughput based on data rate and frame loss rate."""
    # Calculate frame loss rate from retry flag
    total_frames = len(df)
    retry_frames = df['Retry'].sum()
    frame_loss_rate = retry_frames / total_frames
    
    # Calculate throughput for each frame
    df['Throughput'] = df['Data Rate'] * (1 - frame_loss_rate)
    return df

def plot_network_trends(df, output_dir):
    """Create network performance trends plot."""
    colors = ['#2c7da0', '#40916c', '#a8dadc']
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    
    df['Sample Index'] = range(len(df))
    window = len(df) // 20
    
    df['Rolling Data Rate'] = df['Data Rate'].rolling(window=window, min_periods=1).mean()
    df['Rolling Signal'] = df['Signal strength'].rolling(window=window, min_periods=1).mean()
    df['Rolling SNR'] = df['Signal/noise ratio'].rolling(window=window, min_periods=1).mean()
    
    ax1.plot(df['Sample Index'], df['Rolling Data Rate'], 
             color=colors[0], label='Data Rate (Mbps)', linewidth=2)
    ax1.fill_between(df['Sample Index'], 
                     df['Rolling Data Rate'] - df['Data Rate'].std(),
                     df['Rolling Data Rate'] + df['Data Rate'].std(),
                     color=colors[0], alpha=0.2)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['Sample Index'], df['Rolling Signal'],
                  color=colors[1], label='Signal Strength (dBm)', linewidth=2)
    ax1_twin.plot(df['Sample Index'], df['Rolling SNR'],
                  color=colors[2], label='SNR (dB)', linewidth=2)
    
    ax1.set_title('Network Performance Trends', pad=20, fontsize=16)
    ax1.set_xlabel('Sample Number', fontsize=14)
    ax1.set_ylabel('Data Rate (Mbps)', color=colors[0], fontsize=14)
    ax1_twin.set_ylabel('Signal Strength (dBm) / SNR (dB)', color=colors[1], fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'network_trends.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_signal_quality(df, output_dir):
    """Create signal quality distribution plot."""
    colors = ['#2c7da0', '#40916c']
    
    # Create figure with extra space on right for labels
    fig = plt.figure(figsize=(13, 8))
    
    # Create main subplot with adjusted position to leave room for labels
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.85)  # Adjust right margin for labels
    
    sns.violinplot(data=df, y='Signal strength', color=colors[0], alpha=0.3, ax=ax1)
    sns.stripplot(data=df, y='Signal strength', color=colors[0], alpha=0.4, size=3, ax=ax1)
    
    ax1_twin = ax1.twinx()
    sns.violinplot(data=df, y='Signal/noise ratio', color=colors[1], alpha=0.3, ax=ax1_twin)
    sns.stripplot(data=df, y='Signal/noise ratio', color=colors[1], alpha=0.4, size=3, ax=ax1_twin)
    
    ax1.set_title('Signal Quality Distribution', pad=20, fontsize=16)
    ax1.set_ylabel('Signal Strength (dBm)', color=colors[0], fontsize=14)
    ax1_twin.set_ylabel('SNR (dB)', color=colors[1], fontsize=14)
    
    mean_signal = df['Signal strength'].mean()
    mean_snr = df['Signal/noise ratio'].mean()
    ax1.axhline(y=mean_signal, color=colors[0], linestyle='--', alpha=0.8)
    ax1_twin.axhline(y=mean_snr, color=colors[1], linestyle='--', alpha=0.8)
    
    # Add mean value labels in top right corner outside plot
    fig.text(0.87, 0.95, f'Signal Strength Mean:\n{mean_signal:.1f} dBm', 
             color=colors[0], ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    fig.text(0.87, 0.85, f'SNR Mean:\n{mean_snr:.1f} dB', 
             color=colors[1], ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.savefig(output_dir / 'signal_quality.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_mcs_performance(df, output_dir):
    """Create MCS index performance plot."""
    colors = ['#2c7da0', '#40916c']
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    
    df['MCS Index'] = df['MCS Index'].astype(str)
    mcs_stats = df.groupby('MCS Index').agg({
        'Data Rate': ['mean', 'std'],
        'Retry': 'mean'
    }).reset_index()
    
    x = range(len(mcs_stats))
    ax1.plot(x, mcs_stats['Data Rate']['mean'], color=colors[0],
             marker='o', linewidth=2, label='Data Rate')
    ax1.fill_between(x,
                    mcs_stats['Data Rate']['mean'] - mcs_stats['Data Rate']['std'],
                    mcs_stats['Data Rate']['mean'] + mcs_stats['Data Rate']['std'],
                    color=colors[0], alpha=0.2)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, 1 - mcs_stats['Retry']['mean'], color=colors[1],
                  marker='s', linewidth=2, label='Success Rate')
    
    ax1.set_title('MCS Index Performance', pad=20, fontsize=16)
    ax1.set_xlabel('MCS Index', fontsize=14)
    ax1.set_ylabel('Average Data Rate (Mbps)', color=colors[0], fontsize=14)
    ax1_twin.set_ylabel('Success Rate', color=colors[1], fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mcs_stats['MCS Index'])
    ax1.grid(True, alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.savefig(output_dir / 'mcs_performance.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_throughput_analysis(df, output_dir):
    """Create separate throughput analysis plots."""
    plot_network_trends(df, output_dir)
    plot_signal_quality(df, output_dir)
    plot_mcs_performance(df, output_dir)

def plot_throughput_trends(df, output_dir):
    """Plot throughput trends over time."""
    plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1)
    
    # 1. Data Rate Trend
    ax1 = plt.subplot(gs[0])
    df['Sample Index'] = range(len(df))
    sns.lineplot(data=df, x='Sample Index', y='Data Rate', ax=ax1)
    ax1.set_title('Data Rate Trend Over Time')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Data Rate (Mbps)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Signal Strength Trend
    ax2 = plt.subplot(gs[1])
    sns.lineplot(data=df, x='Sample Index', y='Signal strength', ax=ax2)
    ax2.set_title('Signal Strength Trend Over Time')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Signal Strength (dBm)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_throughput_summary(df, output_dir):
    """Generate a summary of throughput statistics."""
    # Calculate basic statistics
    stats = {
        'Total Frames': len(df),
        'Average Throughput (Mbps)': df['Throughput'].mean(),
        'Median Throughput (Mbps)': df['Throughput'].median(),
        'Max Throughput (Mbps)': df['Throughput'].max(),
        'Min Throughput (Mbps)': df['Throughput'].min(),
        'Throughput Std Dev (Mbps)': df['Throughput'].std(),
        'Average Signal Strength (dBm)': df['Signal strength'].mean(),
        'Average SNR (dB)': df['Signal/noise ratio'].mean(),
        'Average Retry Rate (%)': df['Retry'].mean(),
        'Average MCS Index': df['MCS Index'].mean() if 'MCS Index' in df.columns else 'N/A'
    }
    
    # Calculate percentiles
    percentiles = [25, 50, 75, 95]
    for p in percentiles:
        stats[f'Throughput {p}th Percentile (Mbps)'] = df['Throughput'].quantile(p/100)
    
    # Calculate most common values with error handling
    try:
        mcs_mode = df['MCS Index'].mode()
        stats['Most Common MCS Index'] = mcs_mode.iloc[0] if not mcs_mode.empty else 'N/A'
    except:
        stats['Most Common MCS Index'] = 'N/A'
    
    try:
        phy_mode = df['PHY type'].mode()
        stats['Most Common PHY Type'] = phy_mode.iloc[0] if not phy_mode.empty else 'N/A'
    except:
        stats['Most Common PHY Type'] = 'N/A'
    
    # Calculate performance metrics
    stats['Frames with High Throughput (>10 Mbps)'] = (df['Throughput'] > 10).sum()
    stats['Frames with Low Throughput (<1 Mbps)'] = (df['Throughput'] < 1).sum()
    stats['Frames with Weak Signal (<-70 dBm)'] = (df['Signal strength'] < -70).sum()
    stats['Frames with Poor SNR (<10 dB)'] = (df['Signal/noise ratio'] < 10).sum()
    
    # Calculate percentages
    total_frames = len(df)
    stats['High Throughput Percentage'] = (stats['Frames with High Throughput (>10 Mbps)'] / total_frames) * 100
    stats['Low Throughput Percentage'] = (stats['Frames with Low Throughput (<1 Mbps)'] / total_frames) * 100
    stats['Weak Signal Percentage'] = (stats['Frames with Weak Signal (<-70 dBm)'] / total_frames) * 100
    stats['Poor SNR Percentage'] = (stats['Frames with Poor SNR (<10 dB)'] / total_frames) * 100
    
    # Save statistics to file
    with open(output_dir / 'throughput_summary.txt', 'w') as f:
        f.write("WiFi Throughput Analysis Summary\n")
        f.write("===============================\n\n")
        
        # Basic Statistics
        f.write("Basic Statistics:\n")
        f.write("----------------\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Performance Analysis
        f.write("\nPerformance Analysis:\n")
        f.write("-------------------\n")
        f.write(f"High Throughput Frames: {stats['Frames with High Throughput (>10 Mbps)']} ({stats['High Throughput Percentage']:.2f}%)\n")
        f.write(f"Low Throughput Frames: {stats['Frames with Low Throughput (<1 Mbps)']} ({stats['Low Throughput Percentage']:.2f}%)\n")
        f.write(f"Weak Signal Frames: {stats['Frames with Weak Signal (<-70 dBm)']} ({stats['Weak Signal Percentage']:.2f}%)\n")
        f.write(f"Poor SNR Frames: {stats['Frames with Poor SNR (<10 dB)']} ({stats['Poor SNR Percentage']:.2f}%)\n")
        
        # Add diagnosis
        f.write("\nNetwork Diagnosis:\n")
        f.write("----------------\n")
        
        # 1. Signal Quality Assessment
        signal_quality = ""
        avg_signal = df['Signal strength'].mean()
        if avg_signal > -50:
            signal_quality = "Excellent"
        elif avg_signal > -60:
            signal_quality = "Very Good"
        elif avg_signal > -70:
            signal_quality = "Good"
        else:
            signal_quality = "Poor"
            
        signal_range = df['Signal strength'].max() - df['Signal strength'].min()
        
        f.write("\n1. Signal Analysis:\n")
        f.write(f"- Signal Quality: {signal_quality} ({avg_signal:.1f} dBm)\n")
        f.write(f"- Signal Variation: {signal_range:.1f} dB ")
        if signal_range > 40:
            f.write("(High variation indicates significant client mobility or interference)\n")
        else:
            f.write("(Stable signal strength)\n")
            
        # 2. Performance Analysis
        f.write("\n2. Network Performance:\n")
        avg_throughput = df['Throughput'].mean()
        retry_rate = df['Retry'].mean() * 100
        bandwidth = df['Bandwidth'].mode().iloc[0] if 'Bandwidth' in df.columns else 20
        
        # Theoretical max throughput based on PHY type and bandwidth
        phy_type = df['PHY type'].mode().iloc[0]
        theoretical_max = {
            '802.11ac': bandwidth * 8,  # ~8 Mbps per MHz
            '802.11n': bandwidth * 5,   # ~5 Mbps per MHz
            '802.11a': 54,              # Max for 802.11a
            '802.11g': 54,              # Max for 802.11g
            '802.11b': 11               # Max for 802.11b
        }.get(map_phy_type(phy_type), bandwidth * 5)
        
        efficiency = (avg_throughput / theoretical_max) * 100
        
        f.write(f"- Throughput Efficiency: {efficiency:.1f}% of theoretical maximum\n")
        f.write(f"- Current: {avg_throughput:.1f} Mbps / Theoretical Max: {theoretical_max:.1f} Mbps\n")
        f.write(f"- Frame Retry Rate: {retry_rate:.1f}% ")
        if retry_rate > 10:
            f.write("(High - indicates significant interference or connection issues)\n")
        elif retry_rate > 5:
            f.write("(Moderate - some interference present)\n")
        else:
            f.write("(Normal - good connection quality)\n")
            
        # 3. Channel Environment
        f.write("\n3. Channel Environment:\n")
        snr = df['Signal/noise ratio'].mean()
        snr_std = df['Signal/noise ratio'].std()
        noise_level = df['Noise level'].mean()
        
        f.write(f"- Operating on Channel {df['Actual Channel'].mode().iloc[0]}\n")
        f.write(f"- SNR: {snr:.1f} dB ")
        if snr > 40:
            f.write("(Excellent - very clean RF environment)\n")
        elif snr > 25:
            f.write("(Good - minimal interference)\n")
        else:
            f.write("(Poor - significant interference present)\n")
            
        f.write(f"- Noise Floor: {noise_level:.1f} dBm ")
        if noise_level < -90:
            f.write("(Low noise environment)\n")
        elif noise_level < -85:
            f.write("(Moderate noise level)\n")
        else:
            f.write("(High noise level - possible interference sources nearby)\n")
            
        # 4. Overall Assessment
        f.write("\n4. Overall Assessment:\n")
        issues = []
        if signal_range > 40:
            issues.append("high signal variation")
        if retry_rate > 10:
            issues.append("high retry rate")
        if snr < 25:
            issues.append("low SNR")
        if efficiency < 50:
            issues.append("low throughput efficiency")
            
        if not issues:
            f.write("The network is operating optimally with good signal quality, ")
            f.write("efficient throughput, and minimal interference.\n")
        else:
            f.write(f"The network shows suboptimal performance due to {', '.join(issues)}. ")
            f.write("These issues suggest ")
            if "high signal variation" in issues:
                f.write("client mobility or environmental interference, ")
            if "high retry rate" in issues or "low SNR" in issues:
                f.write("RF interference or physical obstacles, ")
            if "low throughput efficiency" in issues:
                f.write("possible network congestion or protocol overhead, ")
            f.write("which are affecting overall network performance.\n")

def plot_device_analysis(df, output_dir):
    """Create enhanced analysis plots for devices connected to the specific AP."""
    # Modern color palette
    colors = ['#2563eb', '#059669', '#7c3aed', '#ea580c', '#0891b2']
    
    # Filter data for your AP (BSS Id starting with "0c")
    ap_data = df[df['BSS Id'].str.startswith('0c', na=False)]
    
    if len(ap_data) == 0:
        print("No data found for the specified AP")
        return
        
    # Group by device (Transmitter address)
    device_stats = ap_data.groupby('Transmitter address').agg({
        'Signal strength': ['mean', 'std', 'count'],
        'Signal/noise ratio': ['mean', 'std'],
        'Data Rate': ['mean', 'std'],
        'Retry': 'mean',
        'PHY type': lambda x: x.mode().iloc[0] if not x.empty else None
    }).reset_index()
    
    device_stats.columns = ['Device', 'Signal Mean', 'Signal Std', 'Count', 
                          'SNR Mean', 'SNR Std', 'Data Rate Mean', 'Data Rate Std', 
                          'Retry Rate', 'PHY Type']
    
    # Create figure with subplots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    gs.update(wspace=0.3, hspace=0.4)
    
    # 1. Device Signal Quality (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    device_stats = device_stats.sort_values('Signal Mean', ascending=True)
    
    y_pos = np.arange(len(device_stats))
    bars = ax1.barh(y_pos, device_stats['Signal Mean'], 
                    xerr=device_stats['Signal Std'],
                    color=colors[0], alpha=0.8)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1f} dBm',
                ha='left', va='center', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{d[:6]} ({c} frames)" for d, c in 
                         zip(device_stats['Device'], device_stats['Count'])],
                        fontsize=10)
    ax1.set_title('Device Signal Quality by Manufacturer', pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Signal Strength (dBm)', fontsize=12)
    
    # 2. Device Performance Matrix (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(device_stats['Data Rate Mean'], 
                         1 - device_stats['Retry Rate'],
                         s=device_stats['Count']/5,  # Adjusted size scaling
                         c=device_stats['SNR Mean'],
                         cmap='viridis',
                         alpha=0.7)
    
    # Add device labels with arrows
    for idx, row in device_stats.iterrows():
        ax2.annotate(row['Device'][:6],  # Show first 6 characters of MAC
                    (row['Data Rate Mean'], 1 - row['Retry Rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Signal-to-Noise Ratio (dB)', fontsize=10)
    
    ax2.set_title('Device Performance by Manufacturer', pad=20, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Data Rate (Mbps)', fontsize=12)
    ax2.set_ylabel('Success Rate', fontsize=12)
    
    # 3. Device Activity Trends (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    ap_data['Sample Index'] = range(len(ap_data))
    window = max(len(ap_data) // 20, 1)  # Ensure window size is at least 1
    
    # Plot top 3 most active devices with enhanced styling
    for i, device in enumerate(device_stats.nlargest(3, 'Count')['Device']):
        device_data = ap_data[ap_data['Transmitter address'] == device]
        rolling_rate = device_data['Data Rate'].rolling(window=window, min_periods=1).mean()
        
        ax3.plot(device_data['Sample Index'], rolling_rate,
                color=colors[i], label=f"{device[:6]} ({len(device_data)} frames)",
                alpha=0.8, linewidth=2)
        
        # Add mean line
        mean_rate = device_data['Data Rate'].mean()
        ax3.axhline(y=mean_rate, color=colors[i], linestyle='--', alpha=0.3)
        ax3.text(len(ap_data), mean_rate, f'Mean: {mean_rate:.0f} Mbps',
                color=colors[i], alpha=0.8, fontsize=8)
    
    ax3.set_title('Device Activity Trends by Manufacturer', pad=20, fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sample Number', fontsize=12)
    ax3.set_ylabel('Data Rate (Mbps)', fontsize=12)
    ax3.legend(title='Manufacturer (frames)', fontsize=9, title_fontsize=10)
    
    # 4. Device Usage Distribution (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create simple pie chart for device usage
    usage_data = device_stats['Count']
    labels = [f"{d[:6]}\n{c:,} frames" for d, c in  # Added thousands separator
              zip(device_stats['Device'], device_stats['Count'])]
    
    # Use a colorblind-friendly palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(device_stats)))
    
    wedges, texts, autotexts = ax4.pie(usage_data, 
                                      labels=labels,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85,
                                      textprops={'fontsize': 10})
    
    # Enhance the appearance of percentage labels
    plt.setp(autotexts, size=9, weight='bold')
    # Add white edges to wedges for better separation
    plt.setp(wedges, edgecolor='white', linewidth=2)
    
    ax4.set_title('Device Usage Distribution', pad=20, fontsize=14, fontweight='bold')
    
    # Save the enhanced visualization
    plt.savefig(output_dir / 'device_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate enhanced device summary
    summary_stats = {
        'Total Devices': len(device_stats),
        'Most Active Device': f"{device_stats.iloc[device_stats['Count'].argmax()]['Device'][:6]} ({device_stats['Count'].max()} frames)",
        'Best Signal Device': f"{device_stats.iloc[device_stats['Signal Mean'].argmax()]['Device'][:6]} ({device_stats['Signal Mean'].max():.1f} dBm)",
        'Best Data Rate Device': f"{device_stats.iloc[device_stats['Data Rate Mean'].argmax()]['Device'][:6]} ({device_stats['Data Rate Mean'].max():.1f} Mbps)",
        'Average Success Rate': f"{(1 - device_stats['Retry Rate'].mean()):.2%}",
        'Dominant PHY Type': device_stats['PHY Type'].mode().iloc[0]
    }
    
    # Save enhanced summary to text file
    with open(output_dir / 'device_summary.txt', 'w') as f:
        f.write("Device Analysis Summary by Manufacturer\n")
        f.write("===================================\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

def plot_throughput_over_time(df, output_dir):
    """Create detailed time-based throughput analysis."""
    plt.figure(figsize=(15, 8))
    
    # Calculate rolling statistics
    window = max(len(df) // 50, 1)  # Ensure window size is at least 1
    df['Rolling Mean'] = df['Throughput'].rolling(window=window, min_periods=1).mean()
    df['Rolling 95th'] = df['Throughput'].rolling(window=window, min_periods=1).quantile(0.95)
    df['Rolling 5th'] = df['Throughput'].rolling(window=window, min_periods=1).quantile(0.05)
    
    # Plot the performance bands
    plt.fill_between(df.index, df['Rolling 5th'], df['Rolling 95th'],
                    alpha=0.2, color='#2563eb', label='5th-95th Percentile Range')
    plt.plot(df.index, df['Rolling Mean'], color='#2563eb', 
            linewidth=2, label='Moving Average')
    
    # Add overall statistics
    mean_throughput = df['Throughput'].mean()
    peak_throughput = df['Throughput'].max()
    plt.axhline(y=mean_throughput, color='red', linestyle='--', 
                label=f'Mean: {mean_throughput:.0f} Mbps')
    plt.axhline(y=peak_throughput, color='green', linestyle='--', 
                label=f'Peak: {peak_throughput:.0f} Mbps')
    
    plt.xlabel('Sample Number', fontsize=12)
    plt.ylabel('Throughput (Mbps)', fontsize=12)
    plt.title('Throughput Performance Over Time', 
             pad=20, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add timestamp markers every 20% of samples
    total_samples = len(df)
    for i in range(5):
        sample_idx = total_samples // 5 * i
        plt.axvline(x=sample_idx, color='gray', alpha=0.3, linestyle=':')
        plt.text(sample_idx, plt.ylim()[0], f'{sample_idx}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_time_analysis.png', dpi=300)
    plt.close()

def plot_throughput_zones(df, output_dir):
    """Create analysis of throughput performance zones."""
    plt.figure(figsize=(12, 8))
    
    # Calculate throughput statistics
    throughput_max = df['Throughput'].max()
    throughput_min = df['Throughput'].min()
    
    # Create dynamic zone boundaries based on actual data
    if throughput_max <= 100:
        zone_boundaries = [throughput_min, 25, 50, 75, throughput_max]
        zone_labels = ['Very Low', 'Low', 'Medium', 'High']
    elif throughput_max <= 300:
        zone_boundaries = [throughput_min, 50, 100, 200, throughput_max]
        zone_labels = ['Basic', 'Standard', 'Good', 'Excellent']
    else:
        # For high throughput networks
        quartiles = [0.25, 0.5, 0.75]
        zone_boundaries = [throughput_min]
        zone_boundaries.extend(df['Throughput'].quantile(quartiles))
        zone_boundaries.append(throughput_max)
        zone_labels = ['Low', 'Medium', 'High', 'Ultra']
    
    # Ensure boundaries are unique and strictly increasing
    zone_boundaries = sorted(list(set(zone_boundaries)))
    
    # Adjust labels if needed
    zone_labels = zone_labels[:len(zone_boundaries)-1]
    
    # Create the zones
    df['Throughput Zone'] = pd.cut(df['Throughput'], 
                                bins=zone_boundaries,
                                labels=zone_labels,
                                include_lowest=True)
    
    # Calculate statistics for each zone
    zone_stats = df.groupby('Throughput Zone', observed=True).agg(
        throughput_count=('Throughput', 'count'),
        throughput_mean=('Throughput', 'mean'),
        throughput_p95=('Throughput', lambda x: x.quantile(0.95)),
        signal_mean=('Signal strength', 'mean'),
        snr_mean=('Signal/noise ratio', 'mean')
    ).round(1)
    
    # Calculate percentage in each zone
    total_frames = len(df)
    percentages = (zone_stats['throughput_count'] / total_frames * 100).round(1)
    
    # Create horizontal bars
    colors = ['#dc2626', '#ea580c', '#16a34a', '#2563eb']
    bars = plt.barh(range(len(percentages)), percentages, color=colors, alpha=0.8)
    
    # Add rich annotations
    for i, (bar, zone) in enumerate(zip(bars, zone_stats.index)):
        # Percentage and count
        count = zone_stats.loc[zone, 'throughput_count']
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {percentages[zone]:.1f}%\n ({count:,} frames)',
                va='center', fontsize=10)
        
        # Zone statistics
        stats_text = (
            f"Avg: {zone_stats.loc[zone, 'throughput_mean']:.0f} Mbps\n"
            f"95th: {zone_stats.loc[zone, 'throughput_p95']:.0f} Mbps\n"
            f"Signal: {zone_stats.loc[zone, 'signal_mean']:.0f} dBm\n"
            f"SNR: {zone_stats.loc[zone, 'snr_mean']:.0f} dB"
        )
        plt.text(1.02, i, stats_text,
                transform=plt.gca().get_yaxis_transform(),
                va='center', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.yticks(range(len(zone_labels)), zone_labels)
    plt.xlabel('Percentage of Frames', fontsize=12)
    plt.title('Throughput Performance Zones Analysis', 
             pad=20, fontsize=14, fontweight='bold')
    
    # Add overall statistics
    overall_stats = (f"Total Frames: {total_frames:,}\n"
                    f"Average Throughput: {df['Throughput'].mean():.0f} Mbps\n"
                    f"Peak Throughput: {df['Throughput'].max():.0f} Mbps")
    plt.text(0.02, 0.02, overall_stats,
             transform=plt.gca().transAxes,
             va='bottom', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_zones.png', dpi=300)
    plt.close()

def plot_signal_throughput_distribution(df, output_dir):
    """Create enhanced signal strength vs throughput distribution analysis."""
    plt.figure(figsize=(15, 10))
    
    # Create signal strength categories
    signal_boundaries = [-90, -75, -65, -55, -45, -35]
    signal_labels = ['Very Weak\n(<-75 dBm)', 'Weak\n(-75 to -65 dBm)', 
                    'Moderate\n(-65 to -55 dBm)', 'Good\n(-55 to -45 dBm)', 
                    'Excellent\n(>-45 dBm)']
    
    df['Signal Category'] = pd.cut(df['Signal strength'], 
                                 bins=signal_boundaries,
                                 labels=signal_labels)
    
    # Calculate statistics for each category
    signal_stats = df.groupby('Signal Category', observed=True).agg(
        throughput_count=('Throughput', 'count'),
        throughput_mean=('Throughput', 'mean'),
        throughput_std=('Throughput', 'std'),
        throughput_p95=('Throughput', lambda x: x.quantile(0.95)),
        signal_mean=('Signal strength', 'mean'),
        signal_std=('Signal strength', 'std')
    )
    
    # Create violin plots with enhanced styling
    valid_categories = [cat for cat in signal_labels if len(df[df['Signal Category'] == cat]) > 0]
    parts = plt.violinplot([df[df['Signal Category'] == cat]['Throughput'].values 
                           for cat in valid_categories],
                          showmeans=True, showextrema=True)
    
    # Customize violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('#2563eb')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    
    # Add statistics annotations
    for i, cat in enumerate(valid_categories):
        stats = signal_stats.loc[cat]
        # Add count and percentile information
        plt.text(i+1, stats['throughput_p95'],
                f"95th: {stats['throughput_p95']:.0f}",
                ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        plt.text(i+1, stats['throughput_mean'],
                f"Mean: {stats['throughput_mean']:.0f}\n"
                f"n={stats['throughput_count']:,}",
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.xticks(range(1, len(valid_categories) + 1), valid_categories)
    plt.xlabel('Signal Strength Category', fontsize=12)
    plt.ylabel('Throughput (Mbps)', fontsize=12)
    plt.title('Throughput Distribution by Signal Strength', 
             pad=20, fontsize=14, fontweight='bold')
    
    # Add correlation coefficient
    correlation = df['Signal strength'].corr(df['Throughput'])
    plt.text(0.02, 0.98,
             f'Signal-Throughput\nCorrelation: {correlation:.2f}',
             transform=plt.gca().transAxes,
             va='top', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'signal_throughput_distribution.png', dpi=300)
    plt.close()

def generate_detailed_statistics(df, output_dir):
    """Generate detailed statistics for all metrics."""
    metrics = ['Throughput', 'Data Rate', 'Signal strength', 'Signal/noise ratio']
    percentiles = [0, 25, 50, 75, 95, 100]
    
    stats_dict = {}
    for metric in metrics:
        stats = df[metric].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        stats_dict[metric] = {
            'Min': stats['min'],
            'Mean': stats['mean'],
            'Median': stats['50%'],
            '75th Percentile': stats['75%'],
            '95th Percentile': stats['95%'],
            'Max': stats['max'],
            'Std Dev': stats['std']
        }
    
    # Calculate correlations
    corr_matrix = df[metrics].corr()
    
    # Save detailed statistics to file
    with open(output_dir / 'detailed_statistics.txt', 'w') as f:
        f.write("Detailed Network Performance Statistics\n")
        f.write("=====================================\n\n")
        
        for metric, stats in stats_dict.items():
            f.write(f"\n{metric} Statistics:\n")
            f.write("-" * (len(metric) + 11) + "\n")
            for stat_name, value in stats.items():
                f.write(f"{stat_name}: {value:.2f}\n")
        
        f.write("\nMetric Correlations:\n")
        f.write("-------------------\n")
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                f.write(f"{metrics[i]} vs {metrics[j]}: {corr_matrix.iloc[i,j]:.2f}\n")

def analyze_wifi_diagnostics(df, output_dir):
    """Perform comprehensive WiFi diagnostics analysis."""
    # Convert MCS Index to numeric, handling any non-numeric values
    df['MCS Index'] = pd.to_numeric(df['MCS Index'], errors='coerce')
    
    # Create figure for diagnostics visualization
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=plt.gcf())
    gs.update(wspace=0.3, hspace=0.4)
    
    # 1. Signal Quality Issues (Top Left)
    ax1 = plt.subplot(gs[0, 0])
    
    # Define signal quality thresholds
    weak_signal = df['Signal strength'] < -70  # dBm
    poor_snr = df['Signal/noise ratio'] < 20   # dB
    
    # Calculate percentages
    weak_signal_pct = (weak_signal.sum() / len(df)) * 100
    poor_snr_pct = (poor_snr.sum() / len(df)) * 100
    
    # Create bar chart
    issues = ['Weak Signal\n(< -70 dBm)', 'Poor SNR\n(< 20 dB)']
    percentages = [weak_signal_pct, poor_snr_pct]
    colors = ['#dc2626', '#ea580c']
    
    bars = ax1.bar(issues, percentages, color=colors, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    ax1.set_title('Signal Quality Issues', pad=20, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Percentage of Frames', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance Bottlenecks (Top Right)
    ax2 = plt.subplot(gs[0, 1])
    
    # Define performance thresholds
    low_throughput = df['Throughput'] < 50  # Mbps
    high_retry = df['Retry'] > 0.1  # 10% retry rate
    low_mcs = df['MCS Index'] < 5
    
    # Calculate percentages
    low_throughput_pct = (low_throughput.sum() / len(df)) * 100
    high_retry_pct = (high_retry.sum() / len(df)) * 100
    low_mcs_pct = (low_mcs.sum() / len(df)) * 100 if not df['MCS Index'].isna().all() else 0
    
    # Create bar chart
    issues = ['Low Throughput\n(< 50 Mbps)', 'High Retry Rate\n(> 10%)', 'Low MCS Index\n(< 5)']
    percentages = [low_throughput_pct, high_retry_pct, low_mcs_pct]
    colors = ['#16a34a', '#dc2626', '#ea580c']
    
    bars = ax2.bar(issues, percentages, color=colors, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    ax2.set_title('Performance Bottlenecks', pad=20, fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage of Frames', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Channel Interference Analysis (Bottom Left)
    ax3 = plt.subplot(gs[1, 0])
    
    # Calculate noise floor statistics
    noise_stats = df.groupby('Actual Channel').agg({
        'Signal/noise ratio': ['mean', 'std', 'count'],
        'Signal strength': ['mean', 'std'],
        'Noise level': 'mean'
    })
    
    # Create scatter plot
    scatter = ax3.scatter(noise_stats.index, 
                         noise_stats[('Signal/noise ratio', 'mean')],
                         s=noise_stats[('Signal/noise ratio', 'count')]/10,
                         c=noise_stats[('Signal/noise ratio', 'std')],
                         cmap='viridis',
                         alpha=0.7)
    
    # Add error bars
    ax3.errorbar(noise_stats.index,
                 noise_stats[('Signal/noise ratio', 'mean')],
                 yerr=noise_stats[('Signal/noise ratio', 'std')],
                 fmt='none', color='gray', alpha=0.3)
    
    # Add annotations for each channel
    for channel in noise_stats.index:
        stats = noise_stats.loc[channel]
        # Add count and SNR information
        ax3.text(channel, stats[('Signal/noise ratio', 'mean')],
                f"SNR: {stats[('Signal/noise ratio', 'mean')]:.0f} dB\n"
                f"n={stats[('Signal/noise ratio', 'count')]:,}",
                ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('SNR Std Dev', fontsize=10)
    
    ax3.set_title('Channel Interference Analysis', pad=20, fontsize=14, fontweight='bold')
    ax3.set_xlabel('Channel', fontsize=12)
    ax3.set_ylabel('Average SNR (dB)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. PHY Type Distribution (Bottom Right)
    ax4 = plt.subplot(gs[1, 1])
    
    # Calculate PHY type distribution
    phy_dist = df['PHY type'].value_counts()
    
    # Create pie chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(phy_dist)))
    wedges, texts, autotexts = ax4.pie(phy_dist.values,
                                      labels=[map_phy_type(phy) for phy in phy_dist.index],
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85)
    
    # Enhance pie chart appearance
    plt.setp(autotexts, size=9, weight='bold')
    plt.setp(wedges, edgecolor='white', linewidth=2)
    
    ax4.set_title('PHY Type Distribution', pad=20, fontsize=14, fontweight='bold')
    
    # Save the diagnostics visualization
    plt.savefig(output_dir / 'wifi_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate diagnostics summary
    diagnostics_summary = {
        'Signal Quality Issues': {
            'Weak Signal Frames': f"{weak_signal_pct:.1f}%",
            'Poor SNR Frames': f"{poor_snr_pct:.1f}%",
            'Average Signal Strength': f"{df['Signal strength'].mean():.1f} dBm",
            'Average SNR': f"{df['Signal/noise ratio'].mean():.1f} dB",
            'Average Noise Level': f"{df['Noise level'].mean():.1f} dBm"
        },
        'Performance Issues': {
            'Low Throughput Frames': f"{low_throughput_pct:.1f}%",
            'High Retry Rate Frames': f"{high_retry_pct:.1f}%",
            'Low MCS Index Frames': f"{low_mcs_pct:.1f}%",
            'Average Throughput': f"{df['Throughput'].mean():.1f} Mbps"
        },
        'Protocol Analysis': {
            'Most Common PHY Type': map_phy_type(df['PHY type'].mode().iloc[0]),
            'Average MCS Index': f"{df['MCS Index'].mean():.1f}" if not df['MCS Index'].isna().all() else "N/A",
            'Retry Rate': f"{(df['Retry'].mean() * 100):.1f}%",
            'Channel Bandwidth': f"{df['Bandwidth'].mode().iloc[0]} MHz" if 'Bandwidth' in df.columns else "N/A"
        },
        'Channel Analysis': {
            'Channel': f"Channel {df['Actual Channel'].mode().iloc[0]}",
            'Channel Count': f"{len(df['Actual Channel'].unique())}",
            'Average SNR': f"{df['Signal/noise ratio'].mean():.1f} dB",
            'SNR Standard Deviation': f"{df['Signal/noise ratio'].std():.1f} dB",
            'Average Signal Strength': f"{df['Signal strength'].mean():.1f} dBm",
            'Signal Strength Range': f"{df['Signal strength'].min():.1f} to {df['Signal strength'].max():.1f} dBm"
        }
    }
    
    # Save diagnostics summary to file
    with open(output_dir / 'wifi_diagnostics_summary.txt', 'w') as f:
        f.write("WiFi Network Diagnostics Summary\n")
        f.write("==============================\n\n")
        
        for category, stats in diagnostics_summary.items():
            f.write(f"\n{category}:\n")
            f.write("-" * len(category) + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        # Add diagnosis
        f.write("\nNetwork Diagnosis:\n")
        f.write("----------------\n")
        
        # 1. Signal Quality Assessment
        signal_quality = ""
        avg_signal = df['Signal strength'].mean()
        if avg_signal > -50:
            signal_quality = "Excellent"
        elif avg_signal > -60:
            signal_quality = "Very Good"
        elif avg_signal > -70:
            signal_quality = "Good"
        else:
            signal_quality = "Poor"
            
        signal_range = df['Signal strength'].max() - df['Signal strength'].min()
        
        f.write("\n1. Signal Analysis:\n")
        f.write(f"- Signal Quality: {signal_quality} ({avg_signal:.1f} dBm)\n")
        f.write(f"- Signal Variation: {signal_range:.1f} dB ")
        if signal_range > 40:
            f.write("(High variation indicates significant client mobility or interference)\n")
        else:
            f.write("(Stable signal strength)\n")
            
        # 2. Performance Analysis
        f.write("\n2. Network Performance:\n")
        avg_throughput = df['Throughput'].mean()
        retry_rate = df['Retry'].mean() * 100
        bandwidth = df['Bandwidth'].mode().iloc[0] if 'Bandwidth' in df.columns else 20
        
        # Theoretical max throughput based on PHY type and bandwidth
        phy_type = df['PHY type'].mode().iloc[0]
        theoretical_max = {
            '802.11ac': bandwidth * 8,  # ~8 Mbps per MHz
            '802.11n': bandwidth * 5,   # ~5 Mbps per MHz
            '802.11a': 54,              # Max for 802.11a
            '802.11g': 54,              # Max for 802.11g
            '802.11b': 11               # Max for 802.11b
        }.get(map_phy_type(phy_type), bandwidth * 5)
        
        efficiency = (avg_throughput / theoretical_max) * 100
        
        f.write(f"- Throughput Efficiency: {efficiency:.1f}% of theoretical maximum\n")
        f.write(f"- Current: {avg_throughput:.1f} Mbps / Theoretical Max: {theoretical_max:.1f} Mbps\n")
        f.write(f"- Frame Retry Rate: {retry_rate:.1f}% ")
        if retry_rate > 10:
            f.write("(High - indicates significant interference or connection issues)\n")
        elif retry_rate > 5:
            f.write("(Moderate - some interference present)\n")
        else:
            f.write("(Normal - good connection quality)\n")
            
        # 3. Channel Environment
        f.write("\n3. Channel Environment:\n")
        snr = df['Signal/noise ratio'].mean()
        snr_std = df['Signal/noise ratio'].std()
        noise_level = df['Noise level'].mean()
        
        f.write(f"- Operating on Channel {df['Actual Channel'].mode().iloc[0]}\n")
        f.write(f"- SNR: {snr:.1f} dB ")
        if snr > 40:
            f.write("(Excellent - very clean RF environment)\n")
        elif snr > 25:
            f.write("(Good - minimal interference)\n")
        else:
            f.write("(Poor - significant interference present)\n")
            
        f.write(f"- Noise Floor: {noise_level:.1f} dBm ")
        if noise_level < -90:
            f.write("(Low noise environment)\n")
        elif noise_level < -85:
            f.write("(Moderate noise level)\n")
        else:
            f.write("(High noise level - possible interference sources nearby)\n")
            
        # 4. Overall Assessment
        f.write("\n4. Overall Assessment:\n")
        issues = []
        if signal_range > 40:
            issues.append("high signal variation")
        if retry_rate > 10:
            issues.append("high retry rate")
        if snr < 25:
            issues.append("low SNR")
        if efficiency < 50:
            issues.append("low throughput efficiency")
            
        if not issues:
            f.write("The network is operating optimally with good signal quality, ")
            f.write("efficient throughput, and minimal interference.\n")
        else:
            f.write(f"The network shows suboptimal performance due to {', '.join(issues)}. ")
            f.write("These issues suggest ")
            if "high signal variation" in issues:
                f.write("client mobility or environmental interference, ")
            if "high retry rate" in issues or "low SNR" in issues:
                f.write("RF interference or physical obstacles, ")
            if "low throughput efficiency" in issues:
                f.write("possible network congestion or protocol overhead, ")
            f.write("which are affecting overall network performance.\n")

def generate_device_performance_summary(df, output_dir):
    """Generate a text-based summary of device performance."""
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Group by device (Transmitter address)
    device_stats = df.groupby('Transmitter address').agg({
        'Signal strength': ['mean', 'std'],
        'Signal/noise ratio': ['mean'],
        'Throughput': ['mean'],
        'Data Rate': ['mean'],
        'Retry': ['mean']
    }).reset_index()

    # Flatten column names
    device_stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in device_stats.columns
    ]

    # Calculate scores and grades
    for idx, row in device_stats.iterrows():
        # Signal Quality Score (40%)
        signal_score = 40 * (min(max(row['Signal strength_mean'], -90), -30) + 90) / 60
        
        # SNR Score (30%)
        snr_score = 30 * min(row['Signal/noise ratio_mean'], 50) / 50
        
        # Stability Score (20%)
        stability_score = 20 * (1 - min(row['Signal strength_std'], 20) / 20)
        
        # Retry Score (10%)
        retry_score = 10 * (1 - min(row['Retry_mean'] * 100, 100) / 100)
        
        # Total Score
        device_stats.loc[idx, 'Connection_Score'] = round(
            signal_score + snr_score + stability_score + retry_score, 2
        )

    # Add grades based on scores
    device_stats['Grade'] = device_stats['Connection_Score'].apply(
        lambda x: 'A+' if x >= 90 else
        'A' if x >= 85 else
        'B+' if x >= 80 else
        'B' if x >= 75 else
        'C+' if x >= 70 else
        'C' if x >= 65 else
        'D+' if x >= 60 else
        'D' if x >= 55 else 'F'
    )

    # Generate the text summary
    with open(output_dir / 'device_connection_quality.txt', 'w') as f:
        f.write("Device Connection Quality Summary\n")
        f.write("===============================\n\n")
        
        # Overall network statistics
        f.write("Network Overview:\n")
        f.write("-----------------\n")
        f.write(f"Total Devices: {len(device_stats)}\n")
        f.write(f"Average Network Score: {device_stats['Connection_Score'].mean():.2f}\n")
        f.write(f"Network Grade Distribution: {dict(device_stats['Grade'].value_counts())}\n\n")
        
        # Device-by-device analysis
        f.write("Device Analysis:\n")
        f.write("---------------\n")
        
        # Sort devices by connection score
        device_stats = device_stats.sort_values('Connection_Score', ascending=False)
        
        for _, device in device_stats.iterrows():
            f.write(f"\nDevice: {device['Transmitter address']}\n")
            f.write(f"Grade: {device['Grade']} (Score: {device['Connection_Score']})\n")
            f.write("Metrics:\n")
            f.write(f"  - Signal Strength: {device['Signal strength_mean']:.1f} dBm ")
            f.write("(Excellent)" if device['Signal strength_mean'] > -50 else 
                   "(Good)" if device['Signal strength_mean'] > -60 else 
                   "(Fair)" if device['Signal strength_mean'] > -70 else "(Poor)")
            f.write(f"\n  - Signal Stability: ±{device['Signal strength_std']:.1f} dB ")
            f.write("(Stable)" if device['Signal strength_std'] < 5 else 
                   "(Moderate)" if device['Signal strength_std'] < 10 else "(Unstable)")
            f.write(f"\n  - SNR: {device['Signal/noise ratio_mean']:.1f} dB ")
            f.write("(Excellent)" if device['Signal/noise ratio_mean'] > 40 else 
                   "(Good)" if device['Signal/noise ratio_mean'] > 25 else 
                   "(Fair)" if device['Signal/noise ratio_mean'] > 15 else "(Poor)")
            f.write(f"\n  - Average Throughput: {device['Throughput_mean']:.1f} Mbps\n")
            f.write(f"  - Average Data Rate: {device['Data Rate_mean']:.1f} Mbps\n")
            f.write(f"  - Retry Rate: {(device['Retry_mean'] * 100):.1f}%\n")
            f.write("-" * 50 + "\n")

        # Add summary recommendations
        f.write("\nConnection Quality Summary:\n")
        f.write("-------------------------\n")
        excellent_devices = len(device_stats[device_stats['Grade'].isin(['A+', 'A'])])
        poor_devices = len(device_stats[device_stats['Grade'].isin(['D+', 'D', 'F'])])
        
        f.write(f"Excellent Connections (A/A+): {excellent_devices}\n")
        f.write(f"Poor Connections (D/F): {poor_devices}\n")
        
        if poor_devices > 0:
            f.write("\nDevices Needing Attention:\n")
            poor_device_list = device_stats[device_stats['Grade'].isin(['D+', 'D', 'F'])]
            for _, device in poor_device_list.iterrows():
                f.write(f"- {device['Transmitter address']} (Grade: {device['Grade']})\n")

def analyze_wifi_throughput(csv_file):
    """Main function to analyze WiFi throughput data and generate visualizations."""
    # Create output directories
    throughput_dir, density_dir = create_visualization_directory()
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Calculate throughput
    df = calculate_throughput(df)
    
    # Generate visualizations and summaries
    plot_network_trends(df, throughput_dir)
    #plot_signal_quality(df, throughput_dir)
    plot_mcs_performance(df, throughput_dir)
    plot_throughput_trends(df, throughput_dir)
    plot_device_analysis(df, throughput_dir)
    
    # Generate new separate correlation analyses
    plot_throughput_over_time(df, throughput_dir)
    plot_throughput_zones(df, throughput_dir)
    plot_signal_throughput_distribution(df, throughput_dir)
    
    # Generate diagnostics analysis
    #analyze_wifi_diagnostics(df, throughput_dir)
    
    #generate_throughput_summary(df, throughput_dir)
    #generate_detailed_statistics(df, throughput_dir)
    
    # Add this line to generate the device performance analysis
    generate_device_performance_summary(df, throughput_dir)
    
    print(f"Throughput analysis complete. Results saved in {throughput_dir}")

if __name__ == "__main__":
    # Get the absolute path of the script's directory
    script_dir = Path(__file__).resolve().parent
    # Get the project root directory (one level up from script_dir)
    project_root = script_dir.parent
    # Use the correct path to the CSV file
    csv_file = project_root / 'results_parser' / 'throughput_analysis.csv'
    analyze_wifi_throughput(csv_file)
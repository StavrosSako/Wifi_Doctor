import os
from pathlib import Path
import subprocess
from typing import Optional, List
from tqdm import tqdm

class WifiAnalyzer:
    def __init__(self):
        # Initialize paths
        self.project_root = Path(__file__).parent.absolute()
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results_parser"
        self.visualization_dir = self.project_root / "visualization_results"
        
        # Script paths
        self.pcap_parser = self.project_root / "pcap_parser" / "new_pcap_parser.py"
        self.density_analyzer = self.project_root / "performance_monitor" / "monitor_density.py"
        self.throughput_analyzer = self.project_root / "performance_monitor" / "monitor_throughput.py"

    def find_pcap_files(self) -> List[Path]:
        """Find all pcap/pcapng files in the data directory."""
        pcap_files = []
        for ext in ['*.pcap', '*.pcapng']:
            pcap_files.extend(list(self.data_dir.glob(ext)))
        return pcap_files

    def display_pcap_files(self, pcap_files: List[Path]) -> Optional[Path]:
        """Display available pcap files and get user selection."""
        if not pcap_files:
            print("No pcap files found in data directory!")
            return None

        print("\nAvailable PCAP files:")
        print("--------------------")
        for idx, file in enumerate(pcap_files, 1):
            print(f"{idx}. {file.name}")

        while True:
            try:
                choice = int(input("\nSelect a file number (or 0 to exit): "))
                if choice == 0:
                    return None
                if 1 <= choice <= len(pcap_files):
                    return pcap_files[choice - 1]
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def get_analysis_mode(self) -> Optional[str]:
        """Get analysis mode from user."""
        print("\nAvailable analysis modes:")
        print("----------------------")
        print("1. Density Analysis")
        print("2. Throughput Analysis")
        print("0. Exit")

        while True:
            try:
                choice = int(input("\nSelect analysis mode: "))
                if choice == 0:
                    return None
                if choice == 1:
                    return "density"
                if choice == 2:
                    return "throughput"
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def get_packet_limit(self) -> int:
        """Get packet limit from user."""
        while True:
            try:
                print("\nEnter packet limit:")
                print("0. Analyze all packets")
                print("n. Limit to first n packets")
                limit = int(input("Enter number: "))
                if limit < 0:
                    print("Please enter a non-negative number.")
                    continue
                return limit if limit > 0 else 5000  # Default to 5000 if 0 is entered
            except ValueError:
                print("Please enter a valid number.")

    def run_pcap_parser(self, pcap_file: Path, mode: str, limit: int) -> bool:
        """Run the PCAP parser with specified parameters."""
        output_file = self.results_dir / f"{mode}_analysis.csv"
        
        print(f"\nParsing {pcap_file.name}...")
        print(f"Mode: {mode}")
        print(f"Packet limit: {limit}")
        
        try:
            cmd = [
                "python3",
                str(self.pcap_parser),
                "-f", str(pcap_file),
                "-o", str(output_file),
                "-m", mode,
                "-l", str(limit)
            ]
            
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error running PCAP parser: {e}")
            return False

    def run_analysis(self, mode: str) -> bool:
        """Run the appropriate analysis script based on mode."""
        try:
            analyzer_script = self.density_analyzer if mode == "density" else self.throughput_analyzer
            
            print(f"\nRunning {mode} analysis...")
            cmd = ["python3", str(analyzer_script)]
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error running analysis: {e}")
            return False

    def run(self):
        """Main execution flow."""
        # Create necessary directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        print("Welcome to WiFi Analyzer!")
        print("=======================")

        # Find available PCAP files
        pcap_files = self.find_pcap_files()
        if not pcap_files:
            print("No PCAP files found in data directory. Please add PCAP files and try again.")
            return

        # Get user selections
        pcap_file = self.display_pcap_files(pcap_files)
        if not pcap_file:
            return

        mode = self.get_analysis_mode()
        if not mode:
            return

        limit = self.get_packet_limit()

        # Run analysis pipeline
        print("\nStarting analysis pipeline...")
        print("=========================")

        # Step 1: Parse PCAP
        if not self.run_pcap_parser(pcap_file, mode, limit):
            print("PCAP parsing failed. Aborting analysis.")
            return

        # Step 2: Run Analysis
        if not self.run_analysis(mode):
            print("Analysis failed.")
            return

        print("\nAnalysis complete!")
        print(f"Results saved in {self.visualization_dir}/{mode}")

def main():
    analyzer = WifiAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main() 
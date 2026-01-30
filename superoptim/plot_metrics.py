import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot metrics distribution")
    parser.add_argument("csv_path", type=str, help="Path to the metrics CSV file")
    parser.add_argument("--output", type=str, default=None, help="Output path for the plot image")
    parser.add_argument("--metric", type=str, default="chamfer-L1", help="Metric to plot")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File {args.csv_path} does not exist.")
        return
        
    print(f"Loading metrics from {args.csv_path}...")
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if args.metric not in df.columns:
        print(f"Error: Metric '{args.metric}' not found in CSV columns: {df.columns.tolist()}")
        return
        
    data = df[args.metric]
    
    # Setup plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Create histogram with KDE
    sns.histplot(data, bins=50, kde=True, color='blue', alpha=0.6)
    
    plt.title(f"Distribution of {args.metric}", fontsize=16)
    plt.xlabel(args.metric, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Add statistics lines
    mean_val = data.mean()
    median_val = data.median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.6f}')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.6f}')
    
    # Add text box with statistics
    stats_text = (
        f"Count: {len(data)}\n"
        f"Mean: {mean_val:.6f}\n"
        f"Median: {median_val:.6f}\n"
        f"Std: {data.std():.6f}\n"
        f"Min: {data.min():.6f}\n"
        f"Max: {data.max():.6f}"
    )
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()
    
    if args.output:
        out_path = args.output
    else:
        out_path = args.csv_path.replace(".csv", f"_{args.metric}_distribution.png")
        
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
    
    print(f"\nStatistics for {args.metric}:")
    print(data.describe())

if __name__ == "__main__":
    main()

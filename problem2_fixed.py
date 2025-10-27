"""
Problem 2: Cluster Usage Analysis
Analyzes cluster usage patterns to understand which clusters are most heavily used over time.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, min as spark_min, max as spark_max, countDistinct, lit
from pyspark.sql.types import TimestampType
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Problem 2: Cluster Usage Analysis')
    parser.add_argument('master', nargs='?', default=None, 
                       help='Spark master URL (e.g., spark://master-ip:7077)')
    parser.add_argument('--net-id', type=str, required=False,
                       help='Your net ID')
    parser.add_argument('--skip-spark', action='store_true',
                       help='Skip Spark processing and regenerate visualizations from existing CSVs')
    return parser.parse_args()

def run_spark_analysis(spark_master=None):
    
    # Initialize Spark Session
    if spark_master:
        spark = SparkSession.builder \
            .appName("Problem2-ClusterUsageAnalysis") \
            .master(spark_master) \
            .getOrCreate()
    else:
        spark = SparkSession.builder \
            .appName("Problem2-ClusterUsageAnalysis") \
            .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Read all log files from S3
    log_path = "s3://rr1303-assignment-spark-cluster-logs/data/application_*/container_*.log"
    print(f"Reading log files from: {log_path}")
    
    try:
        logs_df = spark.read.text(log_path)
    except Exception as e:
        print(f"Error reading log files: {e}")
        spark.stop()
        return False
    
    total_lines = logs_df.count()
    print(f"Total log lines read: {total_lines:,}")
    
    # Extract cluster_id, application_id, and timestamp from log entries
    # Pattern: YY/MM/DD HH:MM:SS LEVEL Component: Message
    timestamp_pattern = r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})'
    
    # Add timestamp column
    logs_with_timestamp = logs_df.withColumn(
        "timestamp_str",
        regexp_extract(col("value"), timestamp_pattern, 1)
    )
    
    # Filter only rows with valid timestamps
    logs_with_timestamp = logs_with_timestamp.filter(col("timestamp_str") != "")
    
    # Get input file name to extract cluster_id and application_id
    from pyspark.sql.functions import input_file_name
    
    logs_with_metadata = logs_with_timestamp.withColumn("file_path", input_file_name())
    
    # Extract cluster_id and app_number from file path
    # Path format: .../application_CLUSTERID_APPNUMBER/container_...log
    cluster_pattern = r'application_(\d+)_(\d+)'
    
    logs_with_ids = logs_with_metadata.withColumn(
        "cluster_id", 
        regexp_extract(col("file_path"), cluster_pattern, 1)
    ).withColumn(
        "app_number",
        regexp_extract(col("file_path"), cluster_pattern, 2)
    )
    
    # Filter valid entries
    logs_with_ids = logs_with_ids.filter(
        (col("cluster_id") != "") & (col("app_number") != "")
    )
    
    # Create application_id
    from pyspark.sql.functions import concat
    logs_with_ids = logs_with_ids.withColumn(
        "application_id",
        concat(lit("application_"), col("cluster_id"), lit("_"), col("app_number"))
    )
    
    # Convert timestamp string to timestamp type
    from pyspark.sql.functions import to_timestamp
    logs_with_ids = logs_with_ids.withColumn(
        "timestamp",
        to_timestamp(col("timestamp_str"), "yy/MM/dd HH:mm:ss")
    )
    
    # Group by application to get start and end times
    app_timeline = logs_with_ids.groupBy("cluster_id", "application_id", "app_number").agg(
        spark_min("timestamp").alias("start_time"),
        spark_max("timestamp").alias("end_time")
    ).orderBy("cluster_id", "app_number")
    
    # Group by cluster to get summary statistics
    cluster_summary = logs_with_ids.groupBy("cluster_id").agg(
        countDistinct("application_id").alias("num_applications"),
        spark_min("timestamp").alias("cluster_first_app"),
        spark_max("timestamp").alias("cluster_last_app")
    ).orderBy(col("num_applications").desc())
    
    # Create output directory
    os.makedirs("data/output", exist_ok=True)
    
    print("Saving timeline data...")
    # OUTPUT 1: Timeline CSV
    app_timeline.coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("data/output/problem2_timeline_temp")
    
    print("Saving cluster summary...")
    # OUTPUT 2: Cluster Summary CSV
    cluster_summary.coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("data/output/problem2_cluster_summary_temp")
    
    # Collect data for statistics
    cluster_data = cluster_summary.collect()
    total_clusters = len(cluster_data)
    total_apps = sum(row['num_applications'] for row in cluster_data)
    avg_apps = total_apps / total_clusters if total_clusters > 0 else 0
    
    # OUTPUT 3: Statistics text file
    print("Generating statistics...")
    stats_lines = []
    stats_lines.append(f"Total unique clusters: {total_clusters}")
    stats_lines.append(f"Total applications: {total_apps}")
    stats_lines.append(f"Average applications per cluster: {avg_apps:.2f}")
    stats_lines.append("")
    stats_lines.append("Most heavily used clusters:")
    
    for row in cluster_data[:10]:  # Top 10 clusters
        cluster_id = row['cluster_id']
        num_apps = row['num_applications']
        stats_lines.append(f"  Cluster {cluster_id}: {num_apps} applications")
    
    stats_text = "\n".join(stats_lines)
    
    with open("data/output/problem2_stats.txt", "w") as f:
        f.write(stats_text)
    
    print("\n" + stats_text)
    
    # Clean up temp directories
    import subprocess
    
    try:
        timeline_file = subprocess.check_output(
            "ls data/output/problem2_timeline_temp/*.csv", 
            shell=True
        ).decode().strip()
        subprocess.run(f"mv {timeline_file} data/output/problem2_timeline.csv", shell=True)
        subprocess.run("rm -rf data/output/problem2_timeline_temp", shell=True)
        print("\n✓ Timeline CSV ready")
    except Exception as e:
        print(f"\nWarning: Could not clean up timeline temp files: {e}")
    
    try:
        summary_file = subprocess.check_output(
            "ls data/output/problem2_cluster_summary_temp/*.csv", 
            shell=True
        ).decode().strip()
        subprocess.run(f"mv {summary_file} data/output/problem2_cluster_summary.csv", shell=True)
        subprocess.run("rm -rf data/output/problem2_cluster_summary_temp", shell=True)
        print("✓ Cluster summary CSV ready")
    except Exception as e:
        print(f"Warning: Could not clean up summary temp files: {e}")
    
    spark.stop()
    return True

def generate_visualizations():
    print("\nGenerating visualizations...")
    
    # Check if required CSV files exist
    timeline_path = "data/output/problem2_timeline.csv"
    summary_path = "data/output/problem2_cluster_summary.csv"
    
    if not os.path.exists(timeline_path):
        print(f"Error: {timeline_path} not found!")
        print("Run without --skip-spark first to generate the data.")
        return False
    
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found!")
        print("Run without --skip-spark first to generate the data.")
        return False
    
    # Read CSV files
    print("Reading CSV files...")
    timeline_df = pd.read_csv(timeline_path)
    summary_df = pd.read_csv(summary_path)
    
    # Convert timestamps
    timeline_df['start_time'] = pd.to_datetime(timeline_df['start_time'])
    timeline_df['end_time'] = pd.to_datetime(timeline_df['end_time'])
    timeline_df['duration_seconds'] = (timeline_df['end_time'] - timeline_df['start_time']).dt.total_seconds()
    
    # Set style
    sns.set_style("whitegrid")
    
    # VISUALIZATION 1: Bar Chart - Applications per Cluster
    print("Creating bar chart...")
    plt.figure(figsize=(12, 6))
    
    # Sort by number of applications
    summary_df_sorted = summary_df.sort_values('num_applications', ascending=False)
    
    # Create bar chart
    bars = plt.bar(range(len(summary_df_sorted)), 
                   summary_df_sorted['num_applications'],
                   color=sns.color_palette("husl", len(summary_df_sorted)))
    
    # Add value labels on top of bars
    for i, (idx, row) in enumerate(summary_df_sorted.iterrows()):
        plt.text(i, row['num_applications'] + 2, 
                str(int(row['num_applications'])),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Applications', fontsize=12, fontweight='bold')
    plt.title('Number of Applications per Cluster', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(summary_df_sorted)), 
               summary_df_sorted['cluster_id'], 
               rotation=45, ha='right')
    plt.tight_layout()
    
    bar_chart_path = "data/output/problem2_bar_chart.png"
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Bar chart saved: {bar_chart_path}")
    
    # VISUALIZATION 2: Density Plot for Largest Cluster
    print("Creating density plot...")
    
    # Find the cluster with most applications
    largest_cluster_id = summary_df_sorted.iloc[0]['cluster_id']
    largest_cluster_data = timeline_df[timeline_df['cluster_id'] == str(largest_cluster_id)]
    
    # Filter out invalid durations
    largest_cluster_data = largest_cluster_data[largest_cluster_data['duration_seconds'] > 0]
    
    n_samples = len(largest_cluster_data)
    
    plt.figure(figsize=(12, 6))
    
    # Create histogram with KDE overlay
    plt.hist(largest_cluster_data['duration_seconds'], 
             bins=30, 
             alpha=0.6, 
             color='skyblue', 
             edgecolor='black',
             density=True,
             label='Histogram')
    
    # Add KDE
    from scipy import stats
    import numpy as np
    if len(largest_cluster_data) > 1:
        density = stats.gaussian_kde(largest_cluster_data['duration_seconds'])
        x_range = np.linspace(largest_cluster_data['duration_seconds'].min(), 
                             largest_cluster_data['duration_seconds'].max(), 
                             200)
        plt.plot(x_range, density(x_range), 'r-', linewidth=2, label='KDE')
    
    plt.xscale('log')
    plt.xlabel('Job Duration (seconds, log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title(f'Job Duration Distribution - Cluster {largest_cluster_id} (n={n_samples})', 
             fontsize=14, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    density_plot_path = "data/output/problem2_density_plot.png"
    plt.savefig(density_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Density plot saved: {density_plot_path}")
    
    return True

def main():
    """Main function."""
    args = parse_arguments()
    
    if args.skip_spark:
        print("⚡ Skipping Spark processing, regenerating visualizations only...")
        success = generate_visualizations()
    else:
        # Run Spark analysis
        success = run_spark_analysis(args.master)
        
        if success:
            # Generate visualizations
            success = generate_visualizations()
    
    if success:
        print("\n" + "="*70)
        print("Problem 2 completed successfully!")
        print("="*70)
        print("\nOutput files created:")
        print("  1. data/output/problem2_timeline.csv")
        print("  2. data/output/problem2_cluster_summary.csv")
        print("  3. data/output/problem2_stats.txt")
        print("  4. data/output/problem2_bar_chart.png")
        print("  5. data/output/problem2_density_plot.png")
    else:
        print("\nProblem 2 failed. Please check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
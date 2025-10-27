"""
Problem 1: Log Level Distribution
Analyzes the distribution of log levels (INFO, WARN, ERROR, DEBUG) across all log files.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, count, lit
import os

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Problem1-LogLevelDistribution") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    # Read all log files
    log_path = "/home/ubuntu/data/raw/application_*/container_*.log"
    
    # Read the log files as text
    logs_df = spark.read.text(log_path)
    
    # Total log lines processed
    total_lines = logs_df.count()
    print(f"Total log lines read: {total_lines:,}")
    
    # Extract log level using regex pattern
    log_level_pattern = r'\s+(INFO|WARN|ERROR|DEBUG)\s+'
    
    logs_with_level = logs_df.withColumn(
        "log_level",
        regexp_extract(col("value"), log_level_pattern, 1)
    )
    
    # Filter out rows where no log level was found 
    logs_with_valid_level = logs_with_level.filter(col("log_level") != "")
    
    # Count lines with valid log levels
    total_with_levels = logs_with_valid_level.count()
    print(f"Total lines with log levels: {total_with_levels:,}")
    
    # OUTPUT 1: Log level counts
    log_level_counts = logs_with_valid_level.groupBy("log_level") \
        .agg(count("*").alias("count")) \
        .orderBy(col("count").desc())
    
    # Create output directory if it doesn't exist
    os.makedirs("data/output", exist_ok=True)
    
    # Write counts to CSV
    log_level_counts.coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("data/output/problem1_counts_temp")
    
    # Move the CSV file to the expected location
    print("Saving problem1_counts.csv...")
    
    # OUTPUT 2: Random sample
    sample_logs = logs_with_valid_level.select(
        col("value").alias("log_entry"),
        col("log_level")
    ).sample(withReplacement=False, fraction=0.01, seed=42).limit(10)
    
    # Write sample to CSV
    sample_logs.coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("data/output/problem1_sample_temp")
    
    print("Saving problem1_sample.csv...")
    
    # OUTPUT 3: Summary statistics
    
    # Collect counts for summary
    counts_data = log_level_counts.collect()
    unique_levels = len(counts_data)
    
    # Build summary text
    summary_lines = []
    summary_lines.append(f"Total log lines processed: {total_lines:,}")
    summary_lines.append(f"Total lines with log levels: {total_with_levels:,}")
    summary_lines.append(f"Unique log levels found: {unique_levels}")
    summary_lines.append("")
    summary_lines.append("Log level distribution:")
    
    for row in counts_data:
        level = row['log_level']
        cnt = row['count']
        percentage = (cnt / total_with_levels) * 100 if total_with_levels > 0 else 0
        summary_lines.append(f"  {level:5s} : {cnt:10,} ({percentage:5.2f}%)")
    
    summary_text = "\n".join(summary_lines)
    
    # Write summary to file
    with open("data/output/problem1_summary.txt", "w") as f:
        f.write(summary_text)
    
    print("Saving problem1_summary.txt...")
    print("\nSummary Preview:")
    print(summary_text)
    # Clean up temporary directories and rename files
    import subprocess
    
    # For counts CSV
    try:
        counts_file = subprocess.check_output(
            "ls data/output/problem1_counts_temp/*.csv", 
            shell=True
        ).decode().strip()
        subprocess.run(f"mv {counts_file} data/output/problem1_counts.csv", shell=True)
        subprocess.run("rm -rf data/output/problem1_counts_temp", shell=True)
    except Exception as e:
        print(f"Warning: Could not clean up counts temp files: {e}")
    
    # For sample CSV
    try:
        sample_file = subprocess.check_output(
            "ls data/output/problem1_sample_temp/*.csv", 
            shell=True
        ).decode().strip()
        subprocess.run(f"mv {sample_file} data/output/problem1_sample.csv", shell=True)
        subprocess.run("rm -rf data/output/problem1_sample_temp", shell=True)
    except Exception as e:
        print(f"Warning: Could not clean up sample temp files: {e}")
    
    print("\n✓ Problem 1 completed successfully!")
    print("Output files created:")
    print("  - data/output/problem1_counts.csv")
    print("  - data/output/problem1_sample.csv")
    print("  - data/output/problem1_summary.txt")
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
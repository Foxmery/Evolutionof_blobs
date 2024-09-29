import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import time

def read_and_process_log(file_path):
    # Replace backslashes with forward slashes to handle Windows file paths correctly
    file_path = file_path.replace("\\", "/")
    print(f"Reading log file from: {file_path}")

    log_data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "Time for" in line:
                    log_data.append(line.strip())
        print("Log file successfully read.")
    except Exception as e:
        print(f"Error reading log file: {e}")
        return pd.DataFrame()  # Return empty DataFrame if file reading fails

    # Parse the log data to extract operation names and times using multiprocessing
    print("Processing log data...")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    operation_times = pool.map(parse_log_entry, log_data)
    pool.close()
    pool.join()

    # Filter out None values in case of parsing errors
    operation_times = [entry for entry in operation_times if entry is not None]

    print(f"Processed {len(operation_times)} log entries.")
    return pd.DataFrame(operation_times, columns=["Operation", "Time (s)"])

def parse_log_entry(entry):
    try:
        parts = entry.split(": ")
        operation = parts[0].replace("Time for ", "")
        time_taken = float(parts[1].split(" ")[0])
        return operation, time_taken
    except Exception as e:
        print(f"Error processing entry: {entry}, Error: {e}")
        return None

def plot_operation_times(df):
    if df.empty:
        print("No data to plot.")
        return

    print("Grouping and plotting data...")
    # Group by operation and calculate total and average times using multiprocessing
    start_time = time.time()
    grouped_df = df.groupby("Operation")["Time (s)"].agg(['sum', 'mean']).reset_index()
    print(f"Data grouping completed in {time.time() - start_time:.2f} seconds.")
    print(f"Number of unique operations to be plotted: {len(grouped_df)}")

    # Plotting the total time taken by each operation
    print("Plotting total time taken by each operation...")
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_df['Operation'], grouped_df['sum'], color='skyblue')
    plt.xlabel('Operation')
    plt.ylabel('Total Time (s)')
    plt.title('Total Time Taken by Each Operation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    print("Total time plot completed.")

    # Plotting the average time taken by each operation
    print("Plotting average time taken by each operation...")
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_df['Operation'], grouped_df['mean'], color='lightgreen')
    plt.xlabel('Operation')
    plt.ylabel('Average Time (s)')
    plt.title('Average Time Taken by Each Operation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    print("Average time plot completed.")

if __name__ == '__main__':
    # Path to the log file
    log_file_path = 'C:\\Users\\vanko\\PycharmProjects\\Evolutionof_blobs\\simulation_log.txt'
    df = read_and_process_log(log_file_path)
    if not df.empty:
        plot_operation_times(df)
    else:
        print("No data available for plotting.")

import os
from ml_logger import logger


def load(folder: str = '2024-12-22'):
    # Directory containing subfolders. Change folder accordingly
    #folder = '2024-12-22'
    root_directory = "./runs/gait-conditioned-agility/" + folder + "/train"

    # Dictionary to store loaded pickle data
    loaded_metrics = {}

    # Iterate over each folder in the root directory
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        
        # Ensure it is a directory
        if os.path.isdir(folder_path):
            metrics_path = os.path.join(folder_path, 'metrics.pkl')
            
            # Check if metrics.pkl exists in the folder
            if os.path.exists(metrics_path):
                # Use folder name as key for dictionary
                variable_name = folder_name.replace('.', '_')  # Replace dot for valid variable naming
                try:
                    # Load the metrics data
                    metrics_data = logger.load_pkl(metrics_path)
                    
                    # Check if the length of the data is 1000
                    if len(metrics_data) == 1000:
                        loaded_metrics[variable_name] = metrics_data
                        print(f"Loaded metrics from: {metrics_path}")
                    else:
                        print(f"Skipping {metrics_path} because length is not 1000")
                except Exception as e:
                    print(f"Error loading {metrics_path}: {e}")
    print(f"Number of Metrics: {len(loaded_metrics)}")
    return loaded_metrics

def single_plot(name: str, loaded_metrics: dict):
    import matplotlib.pyplot as plt

    # Extract iteration numbers and corresponding reward means
    iterations = [entry.get('iterations', 0) for entry in loaded_metrics[name]]
    reward_means = [entry.get('train/episode/rew_total/mean', 0.0) for entry in loaded_metrics[name]]

    # Plotting with enhancements for readability
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, reward_means, color='blue', linewidth=1.5, label='Reward Mean')

    # Axis labels and title
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Train/Episode/Reward Total Mean', fontsize=14)
    plt.title('Reward Mean vs Iterations', fontsize=16)

    # Improve tick visibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Reducing the number of x-tick labels for readability
    tick_interval = max(len(iterations) // 10, 1)  # Show only 10 major ticks
    plt.xticks(iterations[::tick_interval] + [iterations[-1]])

    # Adding a legend
    plt.legend(fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()

def multiple_plot(loaded_metrics: dict):
    import matplotlib.pyplot as plt

    # Initialize a figure
    plt.figure(figsize=(14, 8))

    metric = list(loaded_metrics.keys())[0]
    # Iterate over each loaded set of metrics and plot the mean rewards
    for label, metrics in loaded_metrics.items():
        # Check if there are multiple keys and extract the first one
        first_key = list(metrics.keys())[0] if isinstance(metrics, dict) else None
        data_entries = metrics[first_key] if first_key else metrics  # Use the first key if present
        
        # Extract iteration and reward mean values
        iterations = [entry.get('iterations', 0) for entry in data_entries]
        reward_means = [entry.get('train/episode/rew_total/mean', 0.0) for entry in data_entries]
        
        # Plot the rewards with a label
        plt.plot(iterations, reward_means, label=label, linewidth=1.2)

    # Adding labels, title, grid, and legend
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Train/Episode/Reward Total Mean', fontsize=14)
    plt.title('Mean Total Reward Across All Metrics', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.show()

def multiple_plot_std(loaded_metrics: dict):
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize figure
    plt.figure(figsize=(14, 8))

    # List to store reward means and iterations across all metrics
    all_iterations = []
    all_reward_means = []

    # Iterate over each loaded set of metrics
    for label, metrics in loaded_metrics.items():
        # Extract the first key if it's a dictionary
        first_key = list(metrics.keys())[0] if isinstance(metrics, dict) else None
        data_entries = metrics[first_key] if first_key else metrics  # Use the first key if present
        
        # Extract iterations and reward means for each set of metrics
        iterations = [entry.get('iterations', 0) for entry in data_entries]
        reward_means = [entry.get('train/episode/rew_total/mean', 0.0) for entry in data_entries]
        
        # Append the data to the lists
        all_iterations.append(iterations)
        all_reward_means.append(reward_means)

    # Convert lists to numpy arrays for easier manipulation
    all_iterations = np.array(all_iterations)
    all_reward_means = np.array(all_reward_means)

    # Compute the mean and std along the metrics
    mean_reward = np.mean(all_reward_means, axis=0)
    std_reward = np.std(all_reward_means, axis=0)

    # Compute the mean iterations (assuming they are the same across all metrics)
    mean_iterations = np.mean(all_iterations, axis=0)

    # Plot the mean reward with std shaded region
    plt.plot(mean_iterations, mean_reward, label='Mean Reward', color='b', linewidth=2)
    plt.fill_between(mean_iterations, mean_reward - std_reward, mean_reward + std_reward, color='b', alpha=0.3, label='±1 Std Dev')

    # Adding labels, title, grid, and legend
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Train/Episode/Reward Total Mean', fontsize=14)
    plt.title('Mean Total Reward Across All Metrics with Std Dev', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    loaded_metrics = load(folder = '2024-12-22')
    # single_plot(name = "235534_132056", loaded_metrics=loaded_metrics)
    # multiple_plot(loaded_metrics=loaded_metrics)
    # multiple_plot_std(loaded_metrics=loaded_metrics)
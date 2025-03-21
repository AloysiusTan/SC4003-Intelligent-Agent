import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from mdp_solution import MazeMDP

# Define the exact same color scheme used by the Maze environment
ENV_CMAP = ListedColormap([
    "#f0f0f0",  # 0: Empty
    "#90ee90",  # 1: Goal
    "#cd853f",  # 2: Penalty
    "#404040",  # 3: Wall
    "#1f77b4"   # 4: Start
])

# ----------------------- Dedicated Plotting Functions ----------------------- #
def plot_value_iteration_results(mdp, utilities, policy, maze_name="Maze"):
    """
    Generate side-by-side plots for:
      1) Value Iteration Policy (arrows overlayed on the environment's color scheme).
      2) Value Iteration Utilities (environment color scheme + utility text).
    Saves the image as a PNG file.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

    # -------- LEFT: VI Policy Arrows on Environment Colors --------
    axs[0].imshow(
        mdp.maze,
        cmap=ENV_CMAP,  # Use environment color scheme
        origin='upper',
        extent=[0, mdp.width, mdp.height, 0]
    )
    axs[0].set_title(f"{maze_name}: Value Iteration Policy")
    axs[0].set_xticks(range(mdp.width + 1))
    axs[0].set_yticks(range(mdp.height + 1))
    axs[0].grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    axs[0].set_xlim([0, mdp.width])
    axs[0].set_ylim([mdp.height, 0])

    # Draw arrows for the policy
    for (r, c), act in policy.items():
        if mdp.maze[r, c] == 3:  # skip walls
            continue
        dy, dx = mdp.directions[act]
        x_center = c + 0.5
        y_center = r + 0.5
        axs[0].annotate(
            "",
            xy=(x_center + 0.4 * dx, y_center + 0.4 * dy),
            xytext=(x_center, y_center),
            arrowprops=dict(arrowstyle="->", color='blue', lw=2)
        )

    # Legend for environment colors
    legend_patches = [
        Patch(color='#f0f0f0', label='Empty (-0.05)'),
        Patch(color='#90ee90', label='Goal (+1)'),
        Patch(color='#cd853f', label='Penalty (-1)'),
        Patch(color='#404040', label='Wall'),
        Patch(color='#1f77b4', label='Start')
    ]
    axs[0].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # -------- RIGHT: VI Utilities on Environment Colors --------
    axs[1].imshow(
        mdp.maze,
        cmap=ENV_CMAP,  # Use environment color scheme
        origin='upper',
        extent=[0, mdp.width, mdp.height, 0]
    )
    axs[1].set_title(f"{maze_name}: Value Iteration Utilities")
    axs[1].set_xticks(range(mdp.width + 1))
    axs[1].set_yticks(range(mdp.height + 1))
    axs[1].grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    axs[1].set_xlim([0, mdp.width])
    axs[1].set_ylim([mdp.height, 0])

    # Overlay the utility values
    for (r, c) in mdp.states:
        val = utilities[(r, c)]
        text_color = 'white' if val < 0 else 'black'
        axs[1].text(c + 0.5, r + 0.5, f"{val:.2f}",
                    ha='center', va='center', color=text_color, fontsize=8)

    # Overlay walls if needed
    for i in range(mdp.height):
        for j in range(mdp.width):
            if mdp.maze[i, j] == 3:
                rect = plt.Rectangle((j, i), 1, 1, fill=True, color='#404040')
                axs[1].add_patch(rect)

    plt.tight_layout()
    filename = f"{maze_name.lower()}_value_iteration.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_policy_iteration_results(mdp, utilities, policy, maze_name="Maze"):
    """
    Generate side-by-side plots for:
      1) Policy Iteration Policy (arrows overlayed on the environment's color scheme).
      2) Policy Iteration Utilities (environment color scheme + utility text).
    Saves the image as a PNG file.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

    # -------- LEFT: PI Policy Arrows on Environment Colors --------
    axs[0].imshow(
        mdp.maze,
        cmap=ENV_CMAP,  # Use environment color scheme
        origin='upper',
        extent=[0, mdp.width, mdp.height, 0]
    )
    axs[0].set_title(f"{maze_name}: Policy Iteration Policy")
    axs[0].set_xticks(range(mdp.width + 1))
    axs[0].set_yticks(range(mdp.height + 1))
    axs[0].grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    axs[0].set_xlim([0, mdp.width])
    axs[0].set_ylim([mdp.height, 0])

    for (r, c), act in policy.items():
        if mdp.maze[r, c] == 3:
            continue
        dy, dx = mdp.directions[act]
        x_center = c + 0.5
        y_center = r + 0.5
        axs[0].annotate("",
                        xy=(x_center + 0.4 * dx, y_center + 0.4 * dy),
                        xytext=(x_center, y_center),
                        arrowprops=dict(arrowstyle="->", color='blue', lw=2))

    axs[0].legend(handles=[
        Patch(color='#f0f0f0', label='Empty (-0.05)'),
        Patch(color='#90ee90', label='Goal (+1)'),
        Patch(color='#cd853f', label='Penalty (-1)'),
        Patch(color='#404040', label='Wall'),
        Patch(color='#1f77b4', label='Start')
    ], bbox_to_anchor=(1.05, 1), loc='upper left')

    # -------- RIGHT: PI Utilities on Environment Colors --------
    axs[1].imshow(
        mdp.maze,
        cmap=ENV_CMAP,  # Use environment color scheme
        origin='upper',
        extent=[0, mdp.width, mdp.height, 0]
    )
    axs[1].set_title(f"{maze_name}: Policy Iteration Utilities")
    axs[1].set_xticks(range(mdp.width + 1))
    axs[1].set_yticks(range(mdp.height + 1))
    axs[1].grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    axs[1].set_xlim([0, mdp.width])
    axs[1].set_ylim([mdp.height, 0])

    for (r, c) in mdp.states:
        val = utilities[(r, c)]
        text_color = 'white' if val < 0 else 'black'
        axs[1].text(c + 0.5, r + 0.5, f"{val:.2f}",
                    ha='center', va='center', color=text_color, fontsize=8)

    for i in range(mdp.height):
        for j in range(mdp.width):
            if mdp.maze[i, j] == 3:
                rect = plt.Rectangle((j, i), 1, 1, fill=True, color='#404040')
                axs[1].add_patch(rect)

    plt.tight_layout()
    filename = f"{maze_name.lower()}_policy_iteration.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ----------------------- New Random Maze Generator ----------------------- #
def generate_random_maze(n, m, seed=None, p_white=0.6, p_green=0.1, p_brown=0.1, p_wall=0.2):
    """
    Generate an n x m maze where:
      - 0 = white square (reward -0.05)
      - 1 = green square (reward +1)
      - 2 = brown square (reward -1)
      - 3 = wall (obstacle)
      - 4 = start position (reward -0.05)
    The outer boundary is set to walls (3). Interior cells are randomly assigned
    according to the given probabilities. One random interior cell is set to start (4).
    """
    if seed is not None:
        np.random.seed(seed)
    maze = np.zeros((n, m), dtype=int)
    maze[0, :] = 3
    maze[-1, :] = 3
    maze[:, 0] = 3
    maze[:, -1] = 3
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            maze[i, j] = np.random.choice([0, 1, 2, 3],
                                          p=[p_white, p_green, p_brown, p_wall])
    interior_positions = [(i, j) for i in range(1, n - 1) for j in range(1, m - 1)]
    start_pos = interior_positions[np.random.choice(len(interior_positions))]
    maze[start_pos] = 4
    return maze

# ----------------------- Utility Functions for Custom Metrics ----------------------- #
def compute_vi_custom_metrics(mdp, vi_history, vi_metrics):
    if len(vi_history) >= 2:
        last_util = vi_history[-1]
        prev_util = vi_history[-2]
        abs_diffs = [abs(last_util[s] - prev_util[s]) for s in mdp.states]
        avg_final_change = np.mean(abs_diffs)
        max_final_change = np.max(abs_diffs)
    else:
        avg_final_change = None
        max_final_change = None

    if vi_metrics['iterations'] > 0:
        bellman_backups_per_iteration = vi_metrics['bellman_backups'] / vi_metrics['iterations']
    else:
        bellman_backups_per_iteration = None

    bellman_backups_per_state = vi_metrics['bellman_backups'] / len(mdp.states)
    return avg_final_change, max_final_change, bellman_backups_per_iteration, bellman_backups_per_state

def compute_pi_custom_metrics(mdp, pi_history, eval_iterations):
    total_evaluation_updates = (len(pi_history) - 1) * eval_iterations * len(mdp.states)
    pi_iterations_per_state = (len(pi_history) - 1) / len(mdp.states)
    return total_evaluation_updates, pi_iterations_per_state

# ----------------------- Maze Benchmark Functions ----------------------- #
def run_6x6_maze():
    print("\n=== Running 6x6 Maze Analysis ===")
    maze_layout_6x6 = np.array([
        [1, 3, 1, 0, 0, 1],
        [0, 2, 0, 1, 3, 2],
        [0, 0, 2, 0, 1, 0],
        [0, 0, 4, 2, 0, 1],
        [0, 3, 3, 3, 2, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    mdp = MazeMDP(maze_layout_6x6, discount_factor=0.99)

    # Plot the environment
    plt.figure(figsize=(6, 6))
    mdp.plot_maze(title="6x6 Maze Environment")
    plt.tight_layout()
    plt.savefig('maze_6x6_environment.png', bbox_inches='tight')
    plt.close()

    # Value Iteration
    th = 1e-6
    vi_util, vi_policy, vi_history, vi_metrics = mdp.value_iteration(
        max_iterations=100, threshold=th, return_metrics=True
    )
    print(f"6x6 VI: {vi_metrics['iterations']} iterations, {vi_metrics['time']:.4f} sec, {vi_metrics['bellman_backups']} backups")
    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(vi_history, method="Value Iteration (6x6)")
    plt.tight_layout()
    plt.savefig('6x6_value_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    # Generate side-by-side VI policy and utilities image
    plot_value_iteration_results(mdp, vi_util, vi_policy, maze_name="6x6")
    avg_change, max_change, backups_per_iter, backups_per_state = compute_vi_custom_metrics(mdp, vi_history, vi_metrics)

    # Policy Iteration
    eval_iterations = 20
    start_time = time.time()
    pi_util, pi_policy, pi_history = mdp.policy_iteration(max_iterations=100, eval_iterations=eval_iterations)
    pi_time = time.time() - start_time
    print(f"6x6 PI: {len(pi_history)-1} iterations, {pi_time:.4f} sec")
    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(pi_history, method="Policy Iteration (6x6)")
    plt.tight_layout()
    plt.savefig('6x6_policy_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    # Generate side-by-side PI policy and utilities image
    plot_policy_iteration_results(mdp, pi_util, pi_policy, maze_name="6x6")
    total_pi_updates, pi_iters_per_state = compute_pi_custom_metrics(mdp, pi_history, eval_iterations)

    # Return custom metrics
    metrics = {
        "VI Iterations": vi_metrics['iterations'],
        "VI Time (sec)": vi_metrics['time'],
        "Bellman Backups": vi_metrics['bellman_backups'],
        "Bellman Backups per Iteration": backups_per_iter,
        "Bellman Backups per State": backups_per_state,
        "Avg Final Utility Change": avg_change,
        "Max Final Utility Change": max_change,
        "PI Iterations": len(pi_history) - 1,
        "PI Time (sec)": pi_time,
        "Total PI Evaluation Updates": total_pi_updates,
        "PI Iterations per State": pi_iters_per_state
    }
    return metrics


def run_10x10_maze():
    print("\n=== Running 10x10 Maze Analysis ===")
    maze_layout_10x10 = generate_random_maze(10, 10, seed=42)
    mdp = MazeMDP(maze_layout_10x10, discount_factor=0.99)

    # Plot the environment
    plt.figure(figsize=(6, 6))
    mdp.plot_maze(title="10x10 Maze Environment")
    plt.tight_layout()
    plt.savefig('maze_10x10_environment.png', bbox_inches='tight')
    plt.close()

    # Value Iteration
    th = 1e-6
    vi_util, vi_policy, vi_history, vi_metrics = mdp.value_iteration(
        max_iterations=200, threshold=th, return_metrics=True
    )
    print(f"10x10 VI: {vi_metrics['iterations']} iterations, {vi_metrics['time']:.4f} sec, {vi_metrics['bellman_backups']} backups")
    plot_value_iteration_results(mdp, vi_util, vi_policy, maze_name="10x10")

    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(vi_history, method="Value Iteration (10x10)")
    plt.tight_layout()
    plt.savefig('10x10_value_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    avg_change, max_change, backups_per_iter, backups_per_state = compute_vi_custom_metrics(mdp, vi_history, vi_metrics)

    # Policy Iteration
    eval_iterations = 20
    start_time = time.time()
    pi_util, pi_policy, pi_history = mdp.policy_iteration(max_iterations=100, eval_iterations=eval_iterations)
    pi_time = time.time() - start_time
    print(f"10x10 PI: {len(pi_history)-1} iterations, {pi_time:.4f} sec")
    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(pi_history, method="Policy Iteration (10x10)")
    plt.tight_layout()
    plt.savefig('10x10_policy_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    plot_policy_iteration_results(mdp, pi_util, pi_policy, maze_name="10x10")
    total_pi_updates, pi_iters_per_state = compute_pi_custom_metrics(mdp, pi_history, eval_iterations)

    metrics = {
        "VI Iterations": vi_metrics['iterations'],
        "VI Time (sec)": vi_metrics['time'],
        "Bellman Backups": vi_metrics['bellman_backups'],
        "Bellman Backups per Iteration": backups_per_iter,
        "Bellman Backups per State": backups_per_state,
        "Avg Final Utility Change": avg_change,
        "Max Final Utility Change": max_change,
        "PI Iterations": len(pi_history) - 1,
        "PI Time (sec)": pi_time,
        "Total PI Evaluation Updates": total_pi_updates,
        "PI Iterations per State": pi_iters_per_state
    }
    return metrics


def run_15x15_maze():
    print("\n=== Running 15x15 Maze Analysis ===")
    maze_layout_15x15 = generate_random_maze(15, 15, seed=42)
    mdp = MazeMDP(maze_layout_15x15, discount_factor=0.99)

    # Plot the environment
    plt.figure(figsize=(6, 6))
    mdp.plot_maze(title="15x15 Maze Environment")
    plt.tight_layout()
    plt.savefig('maze_15x15_environment.png', bbox_inches='tight')
    plt.close()

    # Value Iteration
    th = 1e-6
    vi_util, vi_policy, vi_history, vi_metrics = mdp.value_iteration(
        max_iterations=1000, threshold=th, return_metrics=True
    )
    print(f"15x15 VI: {vi_metrics['iterations']} iterations, {vi_metrics['time']:.4f} sec, {vi_metrics['bellman_backups']} backups")
    plot_value_iteration_results(mdp, vi_util, vi_policy, maze_name="15x15")

    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(vi_history, method="Value Iteration (15x15)")
    plt.tight_layout()
    plt.savefig('15x15_value_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    avg_change, max_change, backups_per_iter, backups_per_state = compute_vi_custom_metrics(mdp, vi_history, vi_metrics)

    # Policy Iteration
    eval_iterations = 20
    start_time = time.time()
    pi_util, pi_policy, pi_history = mdp.policy_iteration(max_iterations=100, eval_iterations=eval_iterations)
    pi_time = time.time() - start_time
    print(f"15x15 PI: {len(pi_history)-1} iterations, {pi_time:.4f} sec")

    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(pi_history, method="Policy Iteration (15x15)")
    plt.tight_layout()
    plt.savefig('15x15_policy_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    plot_policy_iteration_results(mdp, pi_util, pi_policy, maze_name="15x15")
    total_pi_updates, pi_iters_per_state = compute_pi_custom_metrics(mdp, pi_history, eval_iterations)

    metrics = {
        "VI Iterations": vi_metrics['iterations'],
        "VI Time (sec)": vi_metrics['time'],
        "Bellman Backups": vi_metrics['bellman_backups'],
        "Bellman Backups per Iteration": backups_per_iter,
        "Bellman Backups per State": backups_per_state,
        "Avg Final Utility Change": avg_change,
        "Max Final Utility Change": max_change,
        "PI Iterations": len(pi_history) - 1,
        "PI Time (sec)": pi_time,
        "Total PI Evaluation Updates": total_pi_updates,
        "PI Iterations per State": pi_iters_per_state
    }
    return metrics


def run_20x20_maze():
    print("\n=== Running 20x20 Maze Analysis ===")
    maze_layout_20x20 = generate_random_maze(20, 20, seed=42)
    mdp = MazeMDP(maze_layout_20x20, discount_factor=0.99)

    # Plot the environment
    plt.figure(figsize=(8, 8))
    mdp.plot_maze(title="20x20 Maze Environment")
    plt.tight_layout()
    plt.savefig('maze_20x20_environment.png', bbox_inches='tight')
    plt.close()

    # Value Iteration
    th = 1e-6
    vi_util, vi_policy, vi_history, vi_metrics = mdp.value_iteration(
        max_iterations=1000, threshold=th, return_metrics=True
    )
    print(f"20x20 VI: {vi_metrics['iterations']} iterations, {vi_metrics['time']:.4f} sec, {vi_metrics['bellman_backups']} backups")
    plot_value_iteration_results(mdp, vi_util, vi_policy, maze_name="20x20")

    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(vi_history, method="Value Iteration (20x20)")
    plt.tight_layout()
    plt.savefig('20x20_value_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    avg_change, max_change, backups_per_iter, backups_per_state = compute_vi_custom_metrics(mdp, vi_history, vi_metrics)

    # Policy Iteration
    eval_iterations = 20
    start_time = time.time()
    pi_util, pi_policy, pi_history = mdp.policy_iteration(max_iterations=100, eval_iterations=eval_iterations)
    pi_time = time.time() - start_time
    print(f"20x20 PI: {len(pi_history)-1} iterations, {pi_time:.4f} sec")

    plt.figure(figsize=(6, 4))
    mdp.plot_utility_convergence(pi_history, method="Policy Iteration (20x20)")
    plt.tight_layout()
    plt.savefig('20x20_policy_iteration_convergence.png', bbox_inches='tight')
    plt.close()

    plot_policy_iteration_results(mdp, pi_util, pi_policy, maze_name="20x20")
    total_pi_updates, pi_iters_per_state = compute_pi_custom_metrics(mdp, pi_history, eval_iterations)

    metrics = {
        "VI Iterations": vi_metrics['iterations'],
        "VI Time (sec)": vi_metrics['time'],
        "Bellman Backups": vi_metrics['bellman_backups'],
        "Bellman Backups per Iteration": backups_per_iter,
        "Bellman Backups per State": backups_per_state,
        "Avg Final Utility Change": avg_change,
        "Max Final Utility Change": max_change,
        "PI Iterations": len(pi_history) - 1,
        "PI Time (sec)": pi_time,
        "Total PI Evaluation Updates": total_pi_updates,
        "PI Iterations per State": pi_iters_per_state
    }
    return metrics

# ----------------------- Benchmark Summary Graph Function ----------------------- #
def plot_benchmark_summary(results):
    sizes = list(results.keys())
    x_labels = sizes
    vi_iterations = [results[size]["VI Iterations"] for size in sizes]
    pi_iterations = [results[size]["PI Iterations"] for size in sizes]
    vi_time = [results[size]["VI Time (sec)"] for size in sizes]
    pi_time = [results[size]["PI Time (sec)"] for size in sizes]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    width = 0.35
    ind = np.arange(len(sizes))

    # Bar chart: Iterations
    axs[0].bar(ind - width/2, vi_iterations, width, label="VI Iterations")
    axs[0].bar(ind + width/2, pi_iterations, width, label="PI Iterations")
    axs[0].set_title("Iterations Comparison")
    axs[0].set_xlabel("Maze Size")
    axs[0].set_ylabel("Iterations")
    axs[0].set_xticks(ind)
    axs[0].set_xticklabels(x_labels)
    axs[0].legend()

    # Bar chart: Time
    axs[1].bar(ind - width/2, vi_time, width, label="VI Time (sec)")
    axs[1].bar(ind + width/2, pi_time, width, label="PI Time (sec)")
    axs[1].set_title("Runtime Comparison")
    axs[1].set_xlabel("Maze Size")
    axs[1].set_ylabel("Time (sec)")
    axs[1].set_xticks(ind)
    axs[1].set_xticklabels(x_labels)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("benchmark_summary.png", bbox_inches="tight")
    plt.close()
    return fig

# ----------------------- Running Benchmarks and Comparison ----------------------- #
if __name__ == "__main__":
    results = {}
    for label, func in [
        ("6x6", run_6x6_maze),
        ("10x10", run_10x10_maze),
        ("15x15", run_15x15_maze),
        ("20x20", run_20x20_maze)
    ]:
        print(f"\n--- Benchmarking {label} Maze ---")
        metrics = func()
        results[label] = metrics
        print(f"\n--- Custom Metrics for {label} Maze ---")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("\n=== Consolidated Benchmark Summary ===")
    for size, metrics in results.items():
        print(f"\n{size} Maze:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    plot_benchmark_summary(results)

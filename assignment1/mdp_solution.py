import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time

class MazeMDP:
    def __init__(self, maze_layout, discount_factor=0.99):
        """
        Initialize the Maze MDP.
        
        Args:
            maze_layout (2D array):
                0 = white square (-0.05 reward)
                1 = green square (+1 reward)
                2 = brown square (-1 reward)
                3 = wall (obstacle)
                4 = start position (-0.05 reward)
            discount_factor (float): discount factor for future rewards
        """
        self.maze = np.array(maze_layout)
        self.height, self.width = self.maze.shape

        self.discount_factor = discount_factor

        # Define actions and directions (Up=0, Right=1, Down=2, Left=3)
        self.actions = [0, 1, 2, 3]
        self.directions = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
        self.action_names = {
            0: "Up",
            1: "Right",
            2: "Down",
            3: "Left"
        }

        # Detect start position (cell = 4)
        start_pos = np.where(self.maze == 4)
        if len(start_pos[0]) == 0:
            raise ValueError("No start position (4) found in maze layout.")
        self.start_state = (start_pos[0][0], start_pos[1][0])

        # Assign rewards
        self.rewards = np.zeros_like(self.maze, dtype=float)
        self.rewards[self.maze == 0] = -0.05  # white squares
        self.rewards[self.maze == 1] = 1.0    # green squares
        self.rewards[self.maze == 2] = -1.0   # brown squares
        self.rewards[self.maze == 4] = -0.05  # start position

        # Collect valid states (not walls)
        self.states = []
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] != 3:  # not a wall
                    self.states.append((i, j))
        self.n_states = len(self.states)

    def get_transition_probs(self, state, action):
        """
        Returns list of (next_state, probability) given current state & action.
        With 0.8 probability move in the chosen direction, and 0.1 each side.
        """
        transitions = []

        # Main direction (80%)
        main_dir = self.directions[action]
        next_state = self._get_next_state(state, main_dir)
        transitions.append((next_state, 0.8))

        # Right turn (10%)
        right_action = (action + 1) % 4
        right_dir = self.directions[right_action]
        right_next_state = self._get_next_state(state, right_dir)
        transitions.append((right_next_state, 0.1))

        # Left turn (10%)
        left_action = (action - 1) % 4
        left_dir = self.directions[left_action]
        left_next_state = self._get_next_state(state, left_dir)
        transitions.append((left_next_state, 0.1))

        return transitions

    def _get_next_state(self, state, direction):
        """
        Returns the next state after attempting to move in 'direction'.
        If the move hits a wall or is out of bounds, remain in current state.
        """
        i, j = state
        di, dj = direction
        ni, nj = i + di, j + dj

        # Check valid (within bounds and not a wall)
        if 0 <= ni < self.height and 0 <= nj < self.width and self.maze[ni, nj] != 3:
            return (ni, nj)
        else:
            return state

    def value_iteration(self, max_iterations=1000, threshold=1e-6, return_metrics=False):
        """
        Value iteration to find optimal utilities & policy.
        Optionally returns performance metrics: iterations, total time, and Bellman backup count.
        """
        utilities = {s: 0.0 for s in self.states}
        policy = {s: 0 for s in self.states}
        utility_history = [utilities.copy()]
        bellman_backups = 0
        start_time = time.time()
        iteration = 0
        while iteration < max_iterations:
            max_change = 0
            new_utilities = utilities.copy()
            for state in self.states:
                bellman_backups += 1  # Count each state update as one backup.
                i, j = state
                action_values = []
                for a in self.actions:
                    transitions = self.get_transition_probs(state, a)
                    exp_util = sum(prob * utilities[s_next] for s_next, prob in transitions)
                    action_values.append(self.rewards[i, j] + self.discount_factor * exp_util)
                best_action_value = max(action_values)
                best_action = np.argmax(action_values)
                new_utilities[state] = best_action_value
                policy[state] = best_action
                max_change = max(max_change, abs(best_action_value - utilities[state]))
            utilities = new_utilities
            utility_history.append(utilities.copy())
            if max_change < threshold:
                break
            iteration += 1

        total_time = time.time() - start_time
        print(f"Value iteration converged after {iteration} iterations (threshold={threshold})")
        if return_metrics:
            metrics = {"iterations": iteration, "time": total_time, "bellman_backups": bellman_backups}
            return utilities, policy, utility_history, metrics
        else:
            return utilities, policy, utility_history

    def policy_iteration(self, max_iterations=100, eval_iterations=20):
        """
        Policy iteration: repeatedly evaluate and improve policy until stable.
        Returns:
            utilities (dict): final utility values
            policy (dict): final policy
            utility_history (list): utilities over iterations
        """
        # Initialize a random policy
        policy = {s: np.random.choice(self.actions) for s in self.states}
        utilities = {s: 0.0 for s in self.states}
        utility_history = [utilities.copy()]

        stable = False
        iteration = 0
        while not stable and iteration < max_iterations:
            # 1) Policy evaluation
            utilities = self._policy_evaluation(policy, utilities, eval_iterations)
            utility_history.append(utilities.copy())

            # 2) Policy improvement
            stable = True
            for state in self.states:
                i, j = state
                old_action = policy[state]

                # Find best action under current utilities
                action_values = []
                for a in self.actions:
                    transitions = self.get_transition_probs(state, a)
                    exp_util = sum(prob * utilities[s_next] for s_next, prob in transitions)
                    action_values.append(self.rewards[i, j] + self.discount_factor * exp_util)

                best_action = np.argmax(action_values)
                if best_action != old_action:
                    policy[state] = best_action
                    stable = False

            iteration += 1

        print(f"Policy iteration converged after {iteration} iterations")
        return utilities, policy, utility_history

    def _policy_evaluation(self, policy, utilities, iterations):
        """
        Iteratively compute utilities for a fixed policy.
        """
        for _ in range(iterations):
            new_utilities = utilities.copy()
            for state in self.states:
                i, j = state
                a = policy[state]
                transitions = self.get_transition_probs(state, a)
                exp_util = sum(prob * utilities[s_next] for s_next, prob in transitions)
                new_utilities[state] = self.rewards[i, j] + self.discount_factor * exp_util
            utilities = new_utilities
        return utilities

    # --------------------------- PLOTTING METHODS --------------------------- #
    def plot_maze(self, ax=None, show_rewards=False, title="Maze Environment"):
        """
        Plot the maze environment with aligned cells and a grid.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Create colormap: white, green, brown, dark gray, blue
        colors = ['#f0f0f0', '#90ee90', '#cd853f', '#404040', '#1f77b4']
        cmap = ListedColormap(colors)

        # Plot using 'extent' to align cells exactly to integer coords
        ax.imshow(self.maze, cmap=cmap, origin='upper',
                extent=[0, self.width, self.height, 0])

        # Set major ticks at each cell boundary and show grid lines.
        ax.set_xticks(np.arange(0, self.width+1, 1))
        ax.set_yticks(np.arange(0, self.height+1, 1))
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)

        ax.set_aspect('equal')
        ax.set_xlim([0, self.width])
        ax.set_ylim([self.height, 0])
        ax.set_title(title, fontsize=14)

        # Add legend
        legend_patches = [
            mpatches.Patch(color=colors[0], label='Empty (-0.05)'),
            mpatches.Patch(color=colors[1], label='Goal (+1)'),
            mpatches.Patch(color=colors[2], label='Penalty (-1)'),
            mpatches.Patch(color=colors[3], label='Wall'),
            mpatches.Patch(color=colors[4], label='Start')
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Optionally overlay rewards
        if show_rewards:
            for i in range(self.height):
                for j in range(self.width):
                    if self.maze[i, j] != 3:  # not a wall
                        ax.text(j+0.5, i+0.5, f"{self.rewards[i, j]:.2f}",
                                ha='center', va='center', color='black', fontsize=8)

        return ax


    def plot_policy(self, policy, ax=None, title="Optimal Policy"):
        """
        Plot arrows indicating the policy action in each cell.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Create colormap for maze (ignore color for start to keep consistent)
        colors = ['#f0f0f0', '#90ee90', '#cd853f', '#404040', '#1f77b4']
        cmap = ListedColormap(colors)

        ax.imshow(self.maze, cmap=cmap, origin='upper',
                extent=[0, self.width, self.height, 0])
        ax.set_aspect('equal')

        # Set major ticks at each cell boundary and show grid lines.
        ax.set_xticks(np.arange(0, self.width+1, 1))
        ax.set_yticks(np.arange(0, self.height+1, 1))
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim([0, self.width])
        ax.set_ylim([self.height, 0])

        # Plot arrows using annotate for better alignment.
        for (i, j), act in policy.items():
            if self.maze[i, j] == 3:  # skip walls
                continue
            dy, dx = self.directions[act]
            start_x = j + 0.5
            start_y = i + 0.5
            end_x = start_x + dx * 0.4
            end_y = start_y + dy * 0.4
            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", color='blue', lw=2))

        # Add legend
        legend_patches = [
            mpatches.Patch(color=colors[0], label='Empty (-0.05)'),
            mpatches.Patch(color=colors[1], label='Goal (+1)'),
            mpatches.Patch(color=colors[2], label='Penalty (-1)'),
            mpatches.Patch(color=colors[3], label='Wall'),
            mpatches.Patch(color=colors[4], label='Start')
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(title, fontsize=14)
        return ax


    def plot_utilities(self, utilities, ax=None, title="State Utilities"):
        """
        Plot a heatmap of utility values, aligned to the maze grid.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Build a matrix of utilities (NaN for walls)
        utility_matrix = np.full_like(self.maze, np.nan, dtype=float)
        for (i, j) in self.states:
            utility_matrix[i, j] = utilities[(i, j)]

        # Plot the matrix
        im = ax.imshow(utility_matrix, cmap='viridis', origin='upper',
                       extent=[0, self.width, self.height, 0])
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Utility')

        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels(np.arange(self.width))
        ax.set_yticklabels(np.arange(self.height))
        ax.set_xlim([0, self.width])
        ax.set_ylim([self.height, 0])

        ax.set_xticks(np.arange(self.width+1), minor=True)
        ax.set_yticks(np.arange(self.height+1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        # Overlay the numeric utility values in the center of each cell
        for (i, j) in self.states:
            val = utilities[(i, j)]
            text_color = 'white' if val < 0 else 'black'
            ax.text(j+0.5, i+0.5, f"{val:.2f}", ha='center', va='center',
                    color=text_color, fontsize=8)

        # Mask walls in dark gray on top
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 3:
                    ax.add_patch(
                        plt.Rectangle((j, i), 1, 1, fill=True, color='#404040', alpha=1.0)
                    )

        ax.set_title(title, fontsize=14)
        return ax

    def plot_utility_convergence(self, utility_history, method="Value Iteration"):
        """
        Plot the utility convergence for all states.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        iterations = range(len(utility_history))

        # Plot each state's utility over iterations
        for s in self.states:
            series = [u[s] for u in utility_history]
            ax.plot(iterations, series, marker='o', markersize=4, label=f"State {s}")

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Utility")
        ax.set_title(f"Utility Convergence for {method} (All States)")
        ax.grid(True)

        # Get handles and labels, then create a horizontal legend below the plot
        handles, labels = ax.get_legend_handles_labels()
        # Adjust 'ncol' to however many columns you want in the legend
        ax.legend(handles, labels,
                loc='upper center',          # put the legend above the 'center' of the axes
                bbox_to_anchor=(0.5, -0.15), # shift the legend box down below the plot
                ncol=6)                      # number of legend columns

        plt.subplots_adjust(bottom=0.25)

        return ax



    def compare_algorithms(self, vi_results, pi_results):
        """
        Compare final utilities & policies from Value Iteration vs Policy Iteration.
        Now includes convergence curves for selected representative states and the average over all states.
        """
        vi_utilities, vi_policy, vi_history = vi_results
        pi_utilities, pi_policy, pi_history = pi_results

        # Compare final utilities
        diff_count = 0
        max_diff = 0
        for s in self.states:
            diff = abs(vi_utilities[s] - pi_utilities[s])
            max_diff = max(max_diff, diff)
            if diff > 1e-3:
                diff_count += 1

        # Compare final policies
        policy_diff_count = sum(vi_policy[s] != pi_policy[s] for s in self.states)

        print(f"Max utility difference: {max_diff:.6f}")
        print(f"States with different utilities: {diff_count} / {len(self.states)}")
        print(f"States with different policies: {policy_diff_count} / {len(self.states)}")

        # Select representative states:
        # - Corners (if valid), start state, highest reward, and lowest reward states.
        corners = [(0, 0), (0, self.width - 1), (self.height - 1, 0), (self.height - 1, self.width - 1)]
        start = self.start_state
        state_rewards = [(s, self.rewards[s[0], s[1]]) for s in self.states]
        high_reward_state = max(state_rewards, key=lambda x: x[1])[0]
        low_reward_state = min(state_rewards, key=lambda x: x[1])[0]

        # Combine unique states for plotting.
        selected_states = list({start, high_reward_state, low_reward_state}.union(set(corners)))

        fig, ax = plt.subplots(figsize=(10, 6))
        iterations_vi = range(len(vi_history))
        iterations_pi = range(len(pi_history))

        for s in selected_states:
            vi_curve = [u[s] for u in vi_history]
            pi_curve = [u[s] for u in pi_history]
            ax.plot(iterations_vi, vi_curve, marker='o', markersize=4, label=f"VI State {s}")
            ax.plot(iterations_pi, pi_curve, marker='x', markersize=4, label=f"PI State {s}")

        # Also plot the average utility convergence over all states.
        avg_vi = [np.mean([u[s] for s in self.states]) for u in vi_history]
        avg_pi = [np.mean([u[s] for s in self.states]) for u in pi_history]
        ax.plot(iterations_vi, avg_vi, marker='s', linestyle='--', label="VI Average")
        ax.plot(iterations_pi, avg_pi, marker='d', linestyle='--', label="PI Average")

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Utility")
        ax.set_title("Algorithm Convergence Comparison (Multiple States and Average)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png')
        plt.close()


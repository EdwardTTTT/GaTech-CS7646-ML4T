""""""                           
"""Assess a betting strategy.                           
                           
Copyright 2018, Georgia Institute of Technology (Georgia Tech)                           
Atlanta, Georgia 30332                           
All Rights Reserved                           
                           
Template code for CS 4646/7646                           
                           
Georgia Tech asserts copyright ownership of this template and all derivative                           
works, including solutions to the projects assigned in this course. Students                           
and other users of this template code are advised not to share it with others                           
or to make it available on publicly viewable websites including repositories                           
such as github and gitlab.  This copyright statement should not be removed                           
or edited.                           
                           
We do grant permission to share solutions privately with non-students such                           
as potential employers. However, sharing with other current or future                           
students of CS 7646 is prohibited and subject to being investigated as a                           
GT honor code violation.                           
                           

-----do not edit anything above this line---

Student Name: Guangqing Tai
GT User ID: gtai3
GT ID: 903968079

This work is a resubmission of CS 7646 Summer 2024 for Fall 2024 term, which may contain partial of previous codes
"""


import numpy as np
import matplotlib.pyplot as plt

def author():
    """Returns the author's GT user ID"""
    return 'gtai3'

def gtid():
    """Returns the author's GT ID"""
    return 903968079

def get_spin_result(win_prob):
    return np.random.random() <= win_prob

def run_episode(win_prob, bankroll=np.inf, max_winnings=80, max_bets=1000):
    winnings = np.zeros(max_bets + 1)
    bet_amount = 1
    episode_winnings = 0

    for i in range(1, max_bets + 1):
        if episode_winnings >= max_winnings or bankroll <= 0:
            winnings[i:] = episode_winnings
            break

        won = get_spin_result(win_prob)

        if won:
            episode_winnings += bet_amount
            bet_amount = 1
        else:
            episode_winnings -= bet_amount
            bet_amount *= 2

        bankroll -= bet_amount if bankroll < bet_amount else 0
        winnings[i] = episode_winnings

    return winnings

def run_experiments(win_prob, num_episodes, bankroll=np.inf):
    results = np.zeros((num_episodes, 1001))
    for i in range(num_episodes, ):
        results[i] = run_episode(win_prob, bankroll)

    return results


def plot_results(results, title, ylabel="Winnings", save_path=None):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, results.shape[0]))  # Use a colormap to get 10 distinct colors

    # Plot each episode with a different color
    for i in range(results.shape[0]):
        plt.plot(results[i], color=colors[i], alpha=0.8, label=f'Episode {i+1}')

    plt.xlabel('Bet Number')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.xlim(0, 300)  # Set X-axis range
    plt.ylim(-256, 100)  # Set Y-axis range
    plt.savefig(save_path)

def plot_mean_std(results, title, save_path=None):
    mean_winnings = np.mean(results, axis=0)
    std_winnings = np.std(results, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_winnings, label="Mean")
    plt.fill_between(range(len(mean_winnings)), mean_winnings - std_winnings, mean_winnings + std_winnings, color='r', alpha=0.2)
    plt.xlabel('Bet Number')
    plt.ylabel('Winnings')
    plt.title(title)
    plt.legend()
    plt.xlim(0, 300)  # Set X-axis range
    plt.ylim(-256, 100)  # Set Y-axis range
    plt.grid(True)
    plt.savefig(save_path)

def plot_median_std(results, title, save_path=None):
    median_winnings = np.median(results, axis=0)
    std_winnings = np.std(results, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(median_winnings, label="Median")
    plt.fill_between(range(len(median_winnings)), median_winnings - std_winnings, median_winnings + std_winnings, color='g', alpha=0.2)
    plt.xlabel('Bet Number')
    plt.ylabel('Winnings')
    plt.title(title)
    plt.legend()
    plt.xlim(0, 300)  # Set X-axis range
    plt.ylim(-256, 100)  # Set Y-axis range
    plt.grid(True)
    plt.savefig(save_path)

def test_code():
    win_prob = 18 / 38  # probability of winning on black in American Roulette
    np.random.seed(gtid())

    # Experiment 1
    results = run_experiments(win_prob, 10)
    plot_results(results, "Experiment 1: 10 Episodes of Simple Simulator", save_path="experiment1_10_episodes.png")

    results = run_experiments(win_prob, 1000)
    plot_mean_std(results, "Experiment 1: Mean Winnings over 1000 Episodes", save_path="experiment1_mean_winnings.png")
    plot_median_std(results, "Experiment 1: Median Winnings over 1000 Episodes", save_path="experiment1_median_winnings.png")

    # Experiment 2
    results = run_experiments(win_prob, 1000, bankroll=256)
    plot_mean_std(results, "Experiment 2: Mean Winnings with $256 Bankroll", save_path="experiment2_mean_winnings.png")
    plot_median_std(results, "Experiment 2: Median Winnings with $256 Bankroll", save_path="experiment2_median_winnings.png")
    test2=np.array(results)
    Over_80=np.sum(test2[:, -1] >=80)
    expected_value=(np.sum(test2[:, -1]))/1000
    print(f"Number of records with final winnings over 80: {Over_80}")
    print(f"Expected value of experiment 2: {expected_value}")
if __name__ == "__main__":
    test_code()

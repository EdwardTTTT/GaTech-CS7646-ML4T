"""
Assess a betting strategy.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab. This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Guangqing Tai
GT User ID: gtai3
GT ID: 903968079
"""

# ===================================================================================
# Useful links

# project 1 Homepage
# http://quantsoftware.gatech.edu/Fall_2019_Project_1:_Martingale

# matplotlib.pyplot API overview
# https://matplotlib.org/3.1.1/api/pyplot_summary.html

# bionomial probability calculator
# https://www.stattrek.com/online-calculator/binomial.aspx

# ===================================================================================
# Libraries and caller functions

import numpy as np
import matplotlib.pyplot as plt

def author():
    """Returns the author's GT user ID"""
    return 'gtai3'

def gtid():
    """Returns the author's GT ID"""
    return 903968079

def get_spin_result(win_prob):
    """
    Simulates a spin of a game with a given win probability.

    Args:
        win_prob (float): Probability of winning the game.
    
    Returns:
        bool: True if win, False otherwise.
    """
    return np.random.random() <= win_prob

def martingale_simulator(win_prob, has_fund, fund):
    """
    Simulates a series of bets using the Martingale strategy.

    Args:
        win_prob (float): Probability of winning the game.
        has_fund (bool): Whether the player has a limited fund.
        fund (int): The amount of money in the fund.
    
    Returns:
        np.ndarray: An array representing the player's winnings over time.
    """
    result_array = np.full((1001), 80)    
    episode_winnings = 0

    count = 0
    while episode_winnings < 80:
        won = False
        bet_amount = 1
        while not won:
            if count >= 1001:
                return result_array
            result_array[count] = episode_winnings
            count += 1
            won = get_spin_result(win_prob)
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2

                if has_fund:
                    if episode_winnings == -fund:
                        result_array[count:] = episode_winnings
                        return result_array

                    if episode_winnings - bet_amount < -fund:
                        bet_amount = fund + episode_winnings
    
    return result_array

def plot_martingale_trials_mean(win_prob, num_trials, fund=None, figure_num=1):
    """
    Runs the simulator multiple times and plots the results.

    Args:
        win_prob (float): Probability of winning the game.
        num_trials (int): Number of trials to run.
        fund (int, optional): The amount of money in the fund. Defaults to None.
        figure_num (int): Figure number for saving the plot.
    """
    result_array = np.zeros((num_trials, 1001))
    for index in range(num_trials):
        curr_episode = martingale_simulator(win_prob, fund is not None, fund)
        result_array[index] = curr_episode

    mean_array = np.mean(result_array, axis=0)
    std = np.std(result_array, axis=0)
    mean_plus_array = mean_array + std
    mean_minus_array = mean_array - std


    plt.figure()
    plt.axis([0, 300, -256, 100])
    title = f"Figure {figure_num} - {num_trials} trials"
    if fund is not None:
        title += f" w/ ${fund} fund"
    else:
        title += " w/ infinite fund"
    plt.title(title)
    plt.xlabel("Number of Spins")
    plt.ylabel("Total Winnings")

    plt.plot(mean_array, label="mean")
    plt.plot(mean_plus_array, label="mean + std", linestyle='--')
    plt.plot(mean_minus_array, label="mean - std", linestyle='--')
    plt.legend()
    plt.savefig(f"figure{figure_num}.png")
    plt.clf()
def plot_martingale_trials_median(win_prob, num_trials, fund=None, figure_num=1):
    """
    Runs the simulator multiple times and plots the results.

    Args:
        win_prob (float): Probability of winning the game.
        num_trials (int): Number of trials to run.
        fund (int, optional): The amount of money in the fund. Defaults to None.
        figure_num (int): Figure number for saving the plot.
    """
    result_array = np.zeros((num_trials, 1001))
    for index in range(num_trials):
        curr_episode = martingale_simulator(win_prob, fund is not None, fund)
        result_array[index] = curr_episode

    std = np.std(result_array, axis=0)
    median_array = np.median(result_array, axis=0)
    median_plus_array = median_array + std
    median_minus_array = median_array - std

    plt.figure()
    plt.axis([0, 300, -256, 100])
    title = f"Figure {figure_num} - {num_trials} trials"
    if fund is not None:
        title += f" w/ ${fund} fund"
    else:
        title += " w/ infinite fund"
    plt.title(title)
    plt.xlabel("Number of Spins")
    plt.ylabel("Total Winnings")

    plt.plot(median_array, label="median", color='orange')
    plt.plot(median_plus_array, label="median + std", linestyle='--', color='purple')
    plt.plot(median_minus_array, label="median - std", linestyle='--', color='purple')
    plt.legend()
    plt.savefig(f"figure{figure_num}.png")
    plt.clf()

def plot_martingale_trials(win_prob):
    """
    Plots the results of multiple trials on the same graph for comparison.

    Args:
        win_prob (float): Probability of winning the game.
    """
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 1 - 10 trials w/ $80 fund")
    plt.xlabel("Number of Spins")
    plt.ylabel("Total Winnings")

    for i in range(10):
        curr_episode = martingale_simulator(win_prob, True, 80)
        plt.plot(curr_episode, label=f'Trial {i+1}')

    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()

def test_martingale_code():
    """Runs the test code to generate the figures."""
    win_prob = 0.42
    np.random.seed(903968079)

    plot_martingale_trials(win_prob)
    plot_martingale_trials_mean(win_prob, 1000, fund=80,figure_num=2)
    plot_martingale_trials_median(win_prob, 1000, fund=80,figure_num=3)
    plot_martingale_trials_mean(win_prob, 1000, fund=256,figure_num=4)
    plot_martingale_trials_median(win_prob, 1000, fund=256,figure_num=5)

if __name__ == "__main__":
    np.set_printoptions(threshold=10000000000000, suppress=True)
    test_martingale_code()
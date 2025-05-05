"""                   
Template for implementing QLearner  (c) 2015 Tucker Balch                   
                   
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
this code is a re-submission of previous term, which may contains similar codes
"""                   
import random as rand
import numpy as np
from collections import deque


class QLearner(object):
    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.Q = np.zeros((num_states, num_actions))

        # Initialize state and action
        self.s = 0
        self.a = 0

        if self.dyna > 0:
            self.experience = deque(maxlen=2000)

    def querysetstate(self, s):
        self.s = s
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s])

        if self.verbose:
            print(f"s = {s}, a = {action}")
        self.a = action
        return action

    def query(self, s_prime, r):
        max_q_s_prime = np.max(self.Q[s_prime])
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + \
                                 self.alpha * (r + self.gamma * max_q_s_prime)


        if self.dyna > 0:
            self.experience.append((self.s, self.a, s_prime, r))
            self.run_dyna_updates()


        self.rar *= self.radr


        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r = {r}")

        # Update state and action
        self.s = s_prime
        self.a = action
        return action

    def run_dyna_updates(self):

        for _ in range(self.dyna):
            s, a, s_prime, r = rand.choice(self.experience)
            max_q_s_prime = np.max(self.Q[s_prime])

            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
                           self.alpha * (r + self.gamma * max_q_s_prime)

    def author(self):
        return 'gtai3'

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")

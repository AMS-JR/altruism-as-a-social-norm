# endowment = 1
# Donations and Aspirations are in range [0, endowment]
# A[i] (Aspiration level): represents the proportion of the endowment they expect to receive when playing as a recipient
# D[i] Donation of player i -> # (in other words, strategies directly determine actions) -> they are the actions
# The update of strategies is performed each time step, after individuals have played one DG
# Donations and Aspirations are in range [0, endowment]
# Individual aspirations A[i] are initialised randomly following a uniform distribution U[0, 1] and
# initial donations are constrained to Di = endowment − A[i]

import numpy as np
import matplotlib.pyplot as plt
import time
import random

def play(N, episodes, time_steps, model, endowment, h, l, transient_time_steps= 10000, epsilon = 0.0 ):
    """
    :param N: number of individuals
    :param episodes: number of rounds
    :param time_steps: number of iterations per episode
    :param model: type of model
    :param endowment: total amount to split -> endowment = 1
    :param h: habituation parameter h ∈ [0, 1]
    :param l: learning rate l ∈ [0, 1]
    :param transient_time_steps:
    :param epsilon: for stochasticity
    :return: average_aspirations, average_donations
    """
    average_aspirations = np.array([])
    average_donations = np.array([])

    for episode in range(episodes):
        print(episode)
        stationary_state = False
        # Aspirations and Donations Space
        A = np.zeros((time_steps+1, N))    #A[i]
        D = np.zeros((time_steps+1, N))    #D[i]
        # Initial aspirations and donations
        A[0] = np.random.uniform(0, 1, N)
        D[0] = endowment - A[0]
        print(A)
        print(D)
        for i in time_steps:
            # pairs_of_individuals[[Dictator,Recipient]]
            pairs_of_individuals = np.arange(N).reshape(N/2,2)  # 1000 individuals
            np.random.shuffle(pairs_of_individuals.flat)
            for j in range(len(pairs_of_individuals)):
                # calulate S[i]

                # update A[i]

                # update D[i]

            # find averages

            # append averages

        return average_aspirations, average_donations



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    N = 1000
    episodes = 100
    time_steps = 1000
    transient_time_steps = 10000
    model = "deterministic"
    endowment = 1
    h = 0.2
    l = 0.2
    average_aspiration, average_donation = play(N,time_steps, model, endowment, h, l, transient_time_steps, 0.01)
    print(average_aspiration, average_donation)
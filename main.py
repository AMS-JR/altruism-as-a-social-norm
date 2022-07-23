# endowment = 1
# Donations and Aspirations are in range [0, endowment]
# A[i] (Aspiration level): represents the proportion of the endowment they expect to receive when playing as a recipient
# D[i] Donation of player i -> # (in other words, strategies directly determine actions) -> they are the actions
# The update of strategies is performed each time step, after individuals have played one DG
# Donations and Aspirations are in range [0, endowment]
# Individual aspirations A[i] are initialised randomly following a uniform distribution U[0, 1] and
# initial donations are constrained to Di = endowment − A[i]
# stimuli s_i is in range [−1, 1]
# donations are bounded

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
    average_aspirations = []
    average_donations = []
    stationary_counter = 0
    stationary_state = False
    for episode in range(episodes):
        print(episode)
        # Aspirations and Donations Space -> can i have just 1D-array of N for each(A and D)? test it to see results are same
        A = np.zeros((time_steps+1, N))
        D = np.zeros((time_steps+1, N))
        # Initial aspirations and donations
        A[0] = np.random.uniform(0, 1, N)
        D[0] = endowment - A[0]
        # print(A)
        print(D)
        for t in range(time_steps):
            # pairs_of_individuals[[Dictator,Recipient]]
            pairs_of_individuals = np.arange(N).reshape(int(N/2),2)  # 1000 individuals
            np.random.shuffle(pairs_of_individuals.flat)

            for i in range(len(pairs_of_individuals)):
                # calulate S_i -> stimuli for recipient
                if( endowment != A[t][pairs_of_individuals[i][1]]):
                    s =  (D[t][pairs_of_individuals[i][0]] - A[t][pairs_of_individuals[i][1]])/(endowment - A[t][pairs_of_individuals[i][1]])
                    if s < -1:
                        s = -1
                    if s > 1:
                        s = 1
                else:
                    s = 0
                # update A[i+1]
                if(s >= 0):
                    change_in_aspiration = ((endowment - A[t][pairs_of_individuals[i][1]]) * l * s)
                    A[t+1][pairs_of_individuals[i][1]] = A[t][pairs_of_individuals[i][1]] + change_in_aspiration
                else:
                    change_in_aspiration = (A[t][pairs_of_individuals[i][1]] * l * s)
                    A[t + 1][pairs_of_individuals[i][1]] = A[t][pairs_of_individuals[i][1]] + change_in_aspiration
                # update D[i+1]
                payoff = (h * D[t][pairs_of_individuals[i][0]])
                D[t+1][pairs_of_individuals[i][1]] = ((1 - h) * D[t][pairs_of_individuals[i][1]]) + payoff
                D[t+1][pairs_of_individuals[i][1]] = max(0, min(D[t+1][pairs_of_individuals[i][1]], (endowment - A[t + 1][pairs_of_individuals[i][1]])))
                #aspirations and donations of dictator individuals are unchanged at the beginning of next time step
                # alternative will be to initialise entire A and D spaces with np.random.uniform(0, 1, N) and remove this part -> explore in testing
                A[t+1][pairs_of_individuals[i][0]] = A[t][pairs_of_individuals[i][0]]
                D[t+1][pairs_of_individuals[i][0]] = D[t][pairs_of_individuals[i][0]]
        stationary_counter += 1
        # Check for Stationary state after each episode to end iterations
        if((stationary_counter)*time_steps >= transient_time_steps):
            # play with the slice and write something better
            increment = 4
            while increment < stationary_counter:
                mean_donations_forward = np.mean(D[int(increment*(time_steps/(stationary_counter))):], axis=0)
                slope = np.max(mean_donations_forward) - np.min(mean_donations_forward)
                print("slope: ", slope)
                stationary_state = slope < 10**(-4)
                increment += 1
                if(stationary_state):
                    break

        if(stationary_state):
            break
        # find averages
        average_aspirations.append(np.mean(A, axis=0))
        average_donations.append(np.mean(D, axis=0))
    f_average_aspirations = np.mean(np.array(average_aspirations), axis=0)
    f_average_donations = np.mean(np.array(average_donations), axis=0)
    return f_average_aspirations, f_average_donations



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    N = 1000
    episodes = 20
    time_steps = 1000
    transient_time_steps = 10000
    model = "deterministic"
    endowment = 1
    h = 0.2
    l = 0.2
    average_aspiration, average_donation = play(N, episodes, time_steps, model, endowment, h, l, transient_time_steps, 0.01)
    print("average_aspiration: ", average_aspiration)
    print(len(average_aspiration))
    print(max(average_aspiration), min(average_aspiration))
    # print("average_donation", average_donation)
    # histograms using matplotlib.pyplot
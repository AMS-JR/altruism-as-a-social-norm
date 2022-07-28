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

def play(N, episodes, time_steps, model, endowment, h, l, transient_time_steps, standard_deviation ):
    """
    :param N: number of individuals
    :param episodes: number of rounds
    :param time_steps: number of iterations per episode
    :param model: type of model
    :param endowment: total amount to split -> endowment = 1
    :param h: habituation parameter h ∈ [0, 1]
    :param l: learning rate l ∈ [0, 1]
    :param transient_time_steps:
    :param standard_deviation: for stochasticity
    :return: average_aspirations, average_donations
    """
    average_aspirations = []
    average_donations = []
    stationary_counter = 0
    stationary_state = False
    free_rider = random.randint(0, N)
    for episode in range(episodes):
        print(f"episode: {episode}")
        # Aspirations and Donations Space -> can i have just 1D-array of N for each(A and D)? test if results are same
        A = np.zeros((time_steps+1, N))
        D = np.zeros((time_steps+1, N))
        # Initial aspirations and donations
        A[0] = np.random.uniform(0, 1, N)
        D[0] = endowment - A[0]
        # print(A)
        # print(D)
        for t in range(time_steps):
            # pairs_of_individuals[[Dictator,Recipient]]
            pairs_of_individuals = np.arange(N).reshape(int(N/2), 2)  # 1000 individuals
            np.random.shuffle(pairs_of_individuals.flat)

            for i in range(len(pairs_of_individuals)):
                if model == "stochastic":
                    # for testing, put noise=standard_deviation as is in the article i believe
                    noise = np.random.normal(0, standard_deviation)
                    D[t][pairs_of_individuals[i][0]] = (1 + noise) * D[t][pairs_of_individuals[i][0]]
                    D[t][pairs_of_individuals[i][1]] = (1 + noise) * D[t][pairs_of_individuals[i][1]]
                elif model == "envious":
                    prob_estimate = np.random.uniform(0.0, high=1.0, size=None)
                    if prob_estimate <= 0.05:
                        # print(f"prob_estimate less ~ Donation: {D[t + 1][pairs_of_individuals[i][1]]}")
                        D[t][pairs_of_individuals[i][0]] = min(D[t][pairs_of_individuals[i][0]], 0.5)
                        D[t][pairs_of_individuals[i][1]] = min(D[t][pairs_of_individuals[i][1]], 0.5)
                elif model == "free-riders":
                    if pairs_of_individuals[i][0] == free_rider:    # Dictator is a free-rider
                        D[t][pairs_of_individuals[i][0]] = 0
                        # print(f"free-riders ~ Dictator: {D[t][pairs_of_individuals[i][0]]}")
                    if pairs_of_individuals[i][1] == free_rider:    # Recipient is a free-rider
                        D[t][pairs_of_individuals[i][1]] = 0
                        # print(f"free-riders ~ Recipient: {D[t+1][pairs_of_individuals[i][1]]}")
                else:
                    pass
                # calculate stimuli for recipient
                if endowment != A[t][pairs_of_individuals[i][1]]:
                    s = (D[t][pairs_of_individuals[i][0]] - A[t][pairs_of_individuals[i][1]])/(endowment - A[t][pairs_of_individuals[i][1]])
                    if s < -1:
                        s = -1
                    if s > 1:
                        s = 1
                else:
                    s = 0
                # update A[i+1]
                if s >= 0:
                    change_in_aspiration = ((endowment - A[t][pairs_of_individuals[i][1]]) * l * s)
                    A[t+1][pairs_of_individuals[i][1]] = A[t][pairs_of_individuals[i][1]] + change_in_aspiration
                else:
                    change_in_aspiration = (A[t][pairs_of_individuals[i][1]] * l * s)
                    A[t+1][pairs_of_individuals[i][1]] = A[t][pairs_of_individuals[i][1]] + change_in_aspiration
                # update D[i+1]
                payoff = (h * D[t][pairs_of_individuals[i][0]])
                D[t+1][pairs_of_individuals[i][1]] = ((1 - h) * D[t][pairs_of_individuals[i][1]]) + payoff
                D[t+1][pairs_of_individuals[i][1]] = max(0, min(D[t+1][pairs_of_individuals[i][1]], (endowment - A[t+1][pairs_of_individuals[i][1]])))
                # ifelifelse
                # aspirations and donations of dictator individuals are unchanged at the beginning of next time step
                A[t+1][pairs_of_individuals[i][0]] = A[t][pairs_of_individuals[i][0]]
                D[t+1][pairs_of_individuals[i][0]] = D[t][pairs_of_individuals[i][0]]
        stationary_counter += 1
        # Check for Stationary state after each episode to end iterations
        if stationary_counter*time_steps >= transient_time_steps:
            # play with the slice and write something better
            print(f"Aspirations: {A}")
            print(f"Donations: {D}")
            increment = 4
            while increment < stationary_counter:
                mean_donations_forward = np.mean(D[int(increment*(time_steps/stationary_counter)):], axis=0)
                slope = np.max(mean_donations_forward) - np.min(mean_donations_forward)
                print("slope: ", slope)
                stationary_state = slope < 10**(-4)
                increment += 1
                if stationary_state:
                    break

        if stationary_state:
            break
        # find averages
        average_aspirations.append(np.mean(A, axis=0))
        average_donations.append(np.mean(D, axis=0))
    f_average_aspirations = np.mean(np.array(average_aspirations), axis=0)
    f_average_donations = np.mean(np.array(average_donations), axis=0)
    return f_average_aspirations, f_average_donations


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ls = np.array([0.2, 0.4, 0.6, 0.8])
    hs = np.array([0.2, 0.4, 0.6, 0.8])
    N = 1000
    episodes = 20
    time_steps = 1000
    transient_time_steps = 10000
    model = "stochastic"
    endowment = 1
    num_runs = 16
    freq_limit = 10
    frequencies_aspirations = []
    frequencies_donations = []

    for j in range(4):
        for k in range(4):
            # for i in range(num_runs):
            print(f"run: {(j+1)} ~ {(k+1)}")
            aspirations_frequency = np.zeros(freq_limit)
            donations_frequency = np.zeros(freq_limit)

            average_aspiration, average_donation = play(N, episodes, time_steps, model, endowment, hs[k], ls[j], transient_time_steps, 0.1)

            for i in range(freq_limit):
                aspirations_frequency[i] = (average_aspiration[(average_aspiration >= (i / freq_limit)) & (average_aspiration < ((i + 1) / freq_limit))].size) / average_aspiration.size
                donations_frequency[i] = (average_donation[(average_donation >= (i / freq_limit)) & (average_donation < ((i + 1) / freq_limit))].size) / average_donation.size
            frequencies_aspirations.append(aspirations_frequency)
            frequencies_donations.append(donations_frequency)

    frequencies_aspirations = np.array(frequencies_aspirations)
    frequencies_donations = np.array(frequencies_donations)
    print("frequencies_aspirations: ", frequencies_aspirations)
    # plotting bars
    x = np.arange(freq_limit)
    y = np.linspace(0.00, 1.00, 5)
    width = 0.35  # the width of the bars
    fig, axs = plt.subplots(4, 4, figsize=(5, 2.7))
    def my_plotter(axs, data1, data2, label, pair):
        """
        A helper function to make a graph.
        """
        width = 0.35  # the width of the bars
        out = axs[pair[0], pair[1]].bar(data1, data2, width, label=label)
        if pair[1] == 0:
            axs[pair[0], pair[1]].set_ylabel('frequency')
            axs[pair[0], pair[1]].set_yticks(y)
        if pair[1] != 0:
            axs[pair[0], pair[1]].yaxis.set_visible(False)
        if pair[0] == 3:
            axs[pair[0], pair[1]].set_xlabel('tenths of endowment')
            axs[pair[0], pair[1]].set_xticks(x)
        if pair[0] != 3:
            axs[pair[0], pair[1]].xaxis.set_visible(False)
        if pair[0] == 3 and pair[1] == 3:
            axs[pair[0], pair[1]].legend()
        axs[pair[0], pair[1]].set_title('l=' + str(ls[pair[0]]) + ' and h=' + str(hs[pair[1]]))
        return out

    i = 0
    for j in range(4):
        for k in range(4):
            rects1 = my_plotter(axs, x - width / 2, frequencies_aspirations[i], "Aspirations", (j, k))
            rects2 = my_plotter(axs, x + width / 2, frequencies_donations[i], "Donations", (j, k))
            i +=1

    plt.show()

# aspirations reflect what they expect to gain from any interaction

# donations reflect what they will receive as recipients

# when the donation is larger (than expected) then the stimulus is positive and leads to higher aspirations in the future and vice versa

# habituation effect -> donations received by recipients has an influence in what they will donate in future

# donations are bounded by aspirations,
# in the sense that subjects cannot exceed their own aspiration level when making a donation, in the absence of noise.

# results are robust against the existence of envious individuals, but they are very much affected
# by the presence of selfish individuals or free-riders

# N: number of individuals

# Dictator decides how to split the endowment

# endowment is the total amount to split among the players.

# the two players iteratively change role in every round/episode

# Each time step, pairs of individuals are randomly chosen among a population of N agents
# and roles (dictator D/recipient R) are randomly assigned.

# The update of strategies is performed each time step, after individuals have played one DG

# we define strategy as the quantity a dictator is going to donate, i.e., the donation Di of player i
# (in other words, strategies directly determine actions)

# only recipients, as a result of the game (dictator decisions), update their strategy to be used
# the next time they play the role of dictator -> update is only for recipient

# A[i] (Aspiration level): represents the proportion of the endowment they expect to receive when playing as a
# recipient

# Each individual i playing R (recipient) receives a stimuli s_i in [−1, 1]
# as a consequence of her dictator’s decisions.

# When the difference between the donation received (payoff π[i]) and her aspiration level is positive,
# recipients receive a positive stimuli, and vice versa according to formula (1)

# dictators never give more than what they expect to receive as
# recipients, thus donations never exceed aspiration formula(4)

# positive stimuli increases Aspiration This effect is moderated by a
# learning rate l ∈ [0, 1] that balances the contribution of past experience. in formula(2)

# An individual adapts the donation she is willing to give when playing as a
# dictator as a consequence of the donation she just received, incorporating an habituation parameter h ∈ [0, 1]
# formula(3)

# Donations and Aspirations are in range [0, endowment]

# the amount kept by a dictator after donating is never lower than her aspiration
# level formula(4)

# Without loss of generality, we will choose endowment = 1 for simplicity hereafter.

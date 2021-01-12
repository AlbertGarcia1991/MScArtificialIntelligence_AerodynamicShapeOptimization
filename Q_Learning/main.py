########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                             INM707 - Garcia Plaza, Albert                                            #
########################################################################################################################
import numpy as np
import random
from utils import *


"""
Initialize R-matrix and Q-matrix as two same-shaped matrix with size 9999X9999. The index (from 0 to 9999) is equivalent
to the NACA 4 digits airfoil. As not all NACA airfoils in the frange from 0000 to 99999 are used on the coursework, 
there are some index that will remain empty along all the training. However, initializing the  matrices with bigger size
than required have been considered easier and more readable rather than matrices with the exact number of NACA airfoils
but no correlation between the matrix index and the specific airfoil. R-matrix is always initialized with zeroes, but 
R-matrix can be either initialized with zeroes or with a given constant.
"""
R_matrix = np.zeros((10000, 10000))

Q_init_value = 5
Q_matrix = np.full((10000, 10000), Q_init_value)

# Initialize other hyperparameters. When the coursework has been done, either LR and GAMMA have been set as tuples with
#   multiple values and two additional for loops have been implemented (before loop on line 33) to compute the solution
#   with all possible parameter combinations.
LR = 1  # learning rate
GAMMA = 1  # discount factor
CL_GOAL = 1.2  # lift coefficient set as goal (finish episode when is reached)
N_EPISODES = 10000   # number of episodes to train the network

steps_count = []

airfoil = NACA_airfoils()  # store all feasible airfoils

for episode in range(N_EPISODES):
    current_state = [random.choice(airfoil)]  # choose randomly the initial state between all allowed

    current_cl = [lookup_cl(current_state[-1])]  # get current state's lift coefficient

    epsilon = 0.9  # set initial epsilon value (epsilon-greedy policy)

    step = 1  # Set initial step and verbose
    print("Episode number ", episode)

    while current_cl[-1] < CL_GOAL:  # while the objective is not reached
        next_states = get_future_states(current_state[-1])  # compute all allowed future states

        e_greedy_prob = random.random()  # generate a random number to choose which policy use (explore vs. exploit)
        if e_greedy_prob <= epsilon:  # Explore
            next_state = random.choice(next_states)
        else:  # Exploit
            Q_next_states = []
            for i in range(len(next_states)):
                Q_next_states.append(Q_matrix[current_state[-1], next_states[i]])
            next_state = next_states[int(np.argmax(Q_next_states))]

        # Update current state and current cl
        current_state.append(next_state)
        current_cl.append(lookup_cl(current_state[-1]))

        # Binary reward
        if current_cl[-1] < current_cl[-2]:  # if cl is smaller on the next state
            reward = -1
        elif current_cl[-1] > current_cl[-2]:  # if cl is greater on the next state
            reward = 1
        else:   # if cl remains the same after the action
            reward = 0

        # Get Q values from the allowed future states
        future_states = get_future_states(current_state[-1])
        Q_future_states = []
        for future_state in future_states:
            Q_future_states.append(Q_matrix[current_state[-1], future_state])

        # Update Q-matrix
        error = reward + GAMMA * max(Q_future_states) - Q_matrix[current_state[-2], current_state[-1]]
        Q_matrix[current_state[-2], current_state[-1]] += LR * error

        # Update epsilon to make the algorithm more greedy every episode
        if epsilon > 0.5:
            epsilon *= 0.999
        else:
            epsilon *= 0.9999

        # Step counter and divergence criteria (if goal not reached after 10,000 steps, terminate current episode)
        step += 1
        if step == 10000:
            break

    steps_count.append(step)  # Store the employed number fo steps at previous episode

    if step == 10000:
        print("Max steps reached without model convergence. End of the current episode.")
    else:
        print("Convergence reached with ", step, " steps.")

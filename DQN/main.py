########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                             INM707 - Garcia Plaza, Albert                                            #
#                Code hacked from https://github.com/unnat5/deep-reinforcement-learning/tree/master/dqn                #
########################################################################################################################
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import body_simulation_NACA


# Declare basic parameters of the DQN algrithm
BATCH_SIZE = 16  # batch size to update memory buffer
UPDATE_EVERY = 4  # update weights every every this number of steps

GAMMA = 0.99  # discount rate
LR = 0.1  # learning rate
EPS_0 = 0.9  # initial epsilon value

GOAL = 1.4  # goal to be reached

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # compute through CUDA if available


class Network(nn.Module):
    """
    Creation of the FNC to predict Q-values given the state -prediction of best value action to apply at every state.
    """
    def __init__(self):
        super(Network, self).__init__()  # inherit attributes from Torch's nn.Module
        self.fcn = nn.Sequential(
            nn.Linear(8, 512),  # from input layer with 8 neurons (state space) to second layer with 512 neurons
            nn.ReLU(),
            nn.Linear(512, 256),  # from second layer with 512 neurons to third layer with 256 neurons
            nn.ReLU(),
            nn.Linear(256, 20)  # from third layer with 256 neurons to output layer with 20 neurons (action space)
        )

    def forward(self, x):
        return self.fcn(x)


class Agent:
    """
    Agent objects contains all the information and procedures to store states, execute actions and update weights. Is
    the backbone of the DQN algorithm.
    """
    def __init__(self):
        self.t_step = 0
        self.fcn = Network().cuda()  # fully-connected network to choose the action (actor)
        self.optimizer = optim.Adam(self.fcn.parameters(), lr=LR)  # optimizer and learning rate for back-propagation
        self.memory = ReplayBuffer()  # initialize memory

    def step(self, state, action, reward, next_step, goal):
        """
        Next step function attribute. Store the current tuple of state-action_chosen-reward_obtained-next_step-goal? on
        memory and update weights if required.
        """
        self.memory.add(state, action, reward, next_step, goal)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY  # update weights every UPDATE_EVERY steps
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience)

    def act(self, state, eps=0):
        """
        Choose (using epsilon-greedy policy) one action and return it (as a vector belonging to the action space).
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.fcn.eval()
        with torch.no_grad():
            action_values = self.fcn(state) # forward pass on local network (actor)
        self.fcn.train()  # network's weights update

        # Action choose following the epsilon-greedy policy
        if random.random() > eps:  # if random generated umber if over epsilon, choose best action
            return np.argmax(action_values.cpu().data.numpy())
        else:  # otherwise, choose randomly
            return random.choice(np.arange(20))

    def learn(self, experiences):
        states, actions, rewards, next_states, goals = experiences
        criterion = nn.MSELoss()
        self.fcn.train()
        predicted_targets = self.fcn(states).gather(1, actions)
        with torch.no_grad():
            labels_next = self.fcn(next_states).detach().max(1)[0].unsqueeze(1)
        labels = rewards + (GAMMA * labels_next * (1 - goals))
        loss = criterion(predicted_targets, labels).to(device)
        with open('log.txt', 'a+') as f:
            line = "Optimization step with loss: " + str(float(loss)) + "\n\n"
            f.write(line)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self):
        self.memory = deque()  # Collection's list (optimized data storing) with specified max length
        self.batch_size = BATCH_SIZE
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "goal"]
                                      )

    def add(self, state, action, reward, next_state, goal):
        """
        Add tuple to the memory.
        """
        add_experience = self.experiences(state, action, reward, next_state, goal)
        self.memory.append(add_experience)

    def sample(self):
        """
        Take a random sample from the memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        goals = torch.from_numpy(np.vstack([e.goal for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, goals

    def __len__(self):
        return len(self.memory)


def train(n_episodes=200, max_t=1000):
    """
    Training process. This function executes the main DQN workflow.
    :param n_episodes: Number maximum of episodes to train the network.
    :param max_t: number maximum of steps in one episode.
    """
    eps = EPS_0
    cl_previous = 0.9  # initial state's lift coefficient

    for i_episode in range(1, n_episodes+1):  # for each episode
        state = utils.initial_state()  # initialize the body (initial shape)

        for t in range(max_t):
            action = agent.act(state, eps)  # choose an action following epsilon-greedy policy

            with open('log.txt', 'a+') as f:  # log creation process
                line = "Episode: " + str(i_episode) + ". Step: " + str(t) + "\nState:  " + str(state) + ". Action: " + str(action)
                f.write(line)

            next_state = utils.apply_action(state, action)  # apply the chosen action to generate the new state

            with open('log.txt', 'a+') as f:  # log creation process
                line = ". Next state: " + str(next_state)
                f.write(line)

            x_points, y_points = utils.create_spline(state)
            cross = utils.intersection_check(x_points, y_points)
            if cross:  # check if the curve cross itself. If so, reward is -10 (rather than ban these shapes)
                reward = -10

            else:  # if the shape is valid
                # Get the current lift coefficient (either simulating or loading from already simulated state)
                cl_current = utils.check_if_exists(next_state)
                if not cl_current:  # if the current state has to be simulated
                    try:
                        cl_current = body_simulation_NACA.Body(next_state).cl  # perform simulation
                        # If wrong cl retrieved (greater than 2 -absolute value), set to 0 and do not store its value
                        if abs(cl_current) > 2:
                            cl_current = 0
                        else:
                            utils.save_simulation(next_state, cl_current)

                    # Sometimes, the simulation may not converge or some error might be prompted (problems of the mesh,
                    #   OS terminal, etc.). When this happens, forget about retrieve the lift coefficient (because
                    #   probably has not been generated), and just set the cl to 0 for this step. Do not store the value
                    #   because if the current state is reached again, the simulation may be correctly performed then.
                    except:
                        cl_current = 0

                with open('log.txt', 'a+') as f:  # log creation process
                    line = ". cl_previous: " + str(cl_previous) + ". cl: " + str(cl_current)
                    f.write(line)

                # Reward assignment procedure, if new cl greater than current -> r=+1, if smaller -> r=-1, otherwise 0
                if cl_current > cl_previous:
                    reward = 1
                elif cl_current < cl_previous:
                    reward = -1
                else:
                    reward = 0

            # If the goal has not been reached, continue with the process. Otherwise, set the goal flag to True
            if cl_current < GOAL:
                goal = False
            else:
                goal = True

            with open('log.txt', 'a+') as f:  # log creation process
                line = ". Reward: " + str(reward) + "\n\n"
                f.write(line)

            agent.step(state, action, reward, next_state, goal)

            # Update state and cl from next to current state
            cl_previous = cl_current
            state = next_state

            # When the goal has been reached, store Torch's weights of the NN
            if goal:
                chkpt_name = 'checkpoint_at_' + str(i_episode) + '_episode.pt'
                torch.save(agent.fcn.state_dict(), chkpt_name)
                break

            if eps > 0.5:
                eps *= 0.99
            else:
                eps *= 0.999


agent = Agent()  # Initialize DQN agent
with open('log.txt', 'w+') as f:  # create new log file (deleting all content if old file exists)
    pass
train()  # run training session

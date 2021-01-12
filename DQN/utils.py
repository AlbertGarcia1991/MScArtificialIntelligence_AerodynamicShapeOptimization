########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                             INM707 - Garcia Plaza, Albert                                            #
########################################################################################################################
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import os


def create_spline(state):
    """
    Given a list with 4 pairs (x, y) location, create and return the cubit spline that passes along the four points and
    also along the anchored points.
    :param state: list with 4 movable points with shape [x1, y1, x2, y2, x3, y3, x4, y4].
    :return x_new: x cooordinales of the spline (201 points).
    :return y_new: y cooordinales of the spline (201 points).
    """
    state = state.reshape((4, 2))  # reshape the input list
    state = np.array([[1., 0.], state[0], state[1], [0, 0], state[2], state[3], [1.0, -0.001]])  # include anchored pts
    tck, u = splprep(state.T, u=None, s=0.0)
    u_new = np.linspace(u.min(), u.max(), 201)
    x_new, y_new = splev(u_new, tck, der=0)
    x_new = normalize(x_new)

    return x_new, y_new


def normalize(x):
    """
    Given the list with x coordinates of the spline, returns the same coordinates but with unitary length.
    :param x: list with x coordinates of the spline.
    :return x_norm: normalized list with max(x)-min(x)=1
    """
    x_min = min(x)
    x_max = max(x)
    x_range = x_max - x_min
    x_norm = (x - x_min) / x_range

    return x_norm


def plot(x, y, title="Airfoil Shape"):
    """
    Plotting function to check the created splines. Used for the report and check the algorithm, not during the
    training process.
    """
    plt.title(title)
    plt.xlim(-1, 2)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('$x/c$')
    plt.ylabel('$y/c$')
    plt.plot(x, y)


def intersection_check(x, y):
    """
    Checks if the generated spline cross itself. If so, returns True, otherwise returns False.
    :param x: x coordinates of the spline to be checked.
    :param y: y coordinates of the spline to be checked.
    :return y: boolean value -True is the spline cross itself, False otherwise.
    """
    index = int(np.argwhere(x == min(x)))
    y_extrados = y[:index]  # points belonging to the upper half of the spline
    y_intrados = y[index+1:]  # points belonging to the bottom half of the spline

    for i in range(len(y_extrados)):
        # If at the same x coordinate, the theoretical bottom point if over the upper one, that means CROSSING curve
        if y_extrados[i] < y_intrados[i]:
            return True
        else:
            return False


def initial_state():
    """
    Generates the starting shape (circle with unitary diameter).
    :return state: 8 elements list corresponding to the initial state.
    """
    point_b = [0.75, 0.5]
    point_c = [0.25, 0.5]
    point_e = [0.25, -0.5]
    point_f = [0.75, -0.5]
    state = np.array([point_b, point_c, point_e, point_f]).flatten()

    return state


def apply_action(state, action):
    """
    Given one state and one action, apply this last to generate the next state vector.
    :param state: current state.
    :param action: action applied (as a vector with 20 elements).
    :return state: new state after the action.
    """
    # Obtain all 4 movable points
    point_b = state.reshape((4, 2))[0]
    point_c = state.reshape((4, 2))[1]
    point_e = state.reshape((4, 2))[2]
    point_f = state.reshape((4, 2))[3]

    if action in [0, 5, 10, 15]:  # these action indexes mean stay at the same position
        movement = [0, 0]
    elif action in [1, 6, 11, 16]:  # these action indexes mean move up
        movement = [0, 0.005]
    elif action in [2, 7, 12, 17]:  # these action indexes mean move down
        movement = [0, -0.005]
    elif action in [3, 8, 13, 18]:  # these action indexes mean move left
        movement = [-0.005, 0]
    else:
        movement = [0.005, 0]  # other indexes mean move right

    # Finally, is selected which movable point will be displaced (first five elements of the action vector are referred
    #   to point_b, next five points to point_c, and so on.
    if action < 5:
        point_b += movement
    elif 5 <= action < 10:
        point_c += movement
    elif 10 <= action < 15:
        point_e += movement
    else:
        point_f += movement

    state = np.array([point_b, point_c, point_e, point_f]).flatten()  # convert the new state on the state vector shape

    return state


def check_if_exists(state):
    """
    Check if the current state have been simulated and stored on the Numpy file.
    :param state: state to be checked.
    :return: if the state has been stored, return its lift coefficient. Otherwise, return False.
    """
    if not os.path.exists('alreadySimulated.npy'):  # if the Numpy file does not exist, return False (first simulation)
        return False

    else:
        entries = np.load('alreadySimulated.npy')  # if the state has been previously stored on the file, returns it cl
        for entry in entries:
            if np.array_equal(state, entry[:-1]):
                return entry[-1]

        return False  # if not, return also False


def save_simulation(state, cl):
    """
    Store the tuple state and lift coefficient on the Numpy file.
    :param state: state vector.
    :param cl: state lift coefficient.
    """
    new_entry = np.array([np.append(state, cl)])  # create the tuple to be stored
    if not os.path.exists('alreadySimulated.npy'):  # if is the first element simulated, create the file
        np.save('alreadySimulated.npy', new_entry)

    else:
        # If the file already exists; open it, read it content and append the new state-cl array.
        old_entries = np.load('alreadySimulated.npy')
        new_entries = np.concatenate((old_entries, new_entry))
        np.save('alreadySimulated.npy', new_entries)

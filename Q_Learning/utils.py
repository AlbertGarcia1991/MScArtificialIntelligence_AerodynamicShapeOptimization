########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                             INM707 - Garcia Plaza, Albert                                            #
########################################################################################################################
import numpy as np
from os import path
import body_simulation_NACA


def lookup_cl(airfoil):
    """
    Perform the simulation of the given airfoil and return its lift coefficient. If the current airfoil has been already
    simulated, first check if it is stored on the storing .txt file and retrieve if cl if so.
    :param airfoil: NACA 4-digit name of the airfoil to get its lift coefficient.
    :return cl: lift coefficent.
    """
    # Open the .txt file where already done simulations are stored
    if not path.exists('simulatedAirfoils.txt'):  # if the file does not exist, create it
        with open('simulatedAirfoils.txt', 'w+') as f:
            pass
    with open('simulatedAirfoils.txt', 'r') as f:
        lines = f.read().split('\n')  # read the file's content

    lines = [line for line in lines if line]  # store one non-blank lines

    stored_airfoil = []
    stored_cl = []
    for line in lines:  # each line stores one airfoil's simulation. Lift coefficient retrieved by its position on str
        line = line.split(',')
        stored_airfoil.append(int(line[0]))  # store the NACA-4digit name
        stored_cl.append(float(line[1]))  # store the lift coefficient

    # Convert them to Numpy arrays to future manipulations
    stored_airfoil = np.array(stored_airfoil)
    stored_cl = np.array(stored_cl)

    # If the passed NACA 4-digit airfoil appears on the file, retrieve its lift coefficient
    if airfoil in stored_airfoil:
        index = int(np.argwhere(stored_airfoil == airfoil))
        cl = stored_cl[index]
        return cl

    # Otherwise, perform the simulation and store the coefficient on the .txt file (and return the cl)
    else:
        simulation = True
        counter = 0
        while simulation:
            sim = body_simulation_NACA.Body(airfoil)
            cl = float(sim.cl)
            # Sometimes, the CFD converges to wrong solutions. Lift coefficient of simple single bodies are always has
            #   an upper bound of 2. So, if the simulated coefficient is greater than 2, the simulation is wrong and
            #   is repeated (4 times more, if still gives wrong results, set the cl as 0 and continue).
            if abs(cl) < 2:
                simulation = False
            else:
                if counter > 5:
                    cl = 0
                    simulation = False

        new_line = str(airfoil) + ',' + str(cl) + '\n' # new line to store on .txt file
        with open('simulatedAirfoils.txt', 'a') as f:
            f.write(new_line)
        return cl


def get_future_states(airfoil):
    """
    Compute all feasible states allowed where end from the given state (in other words, all allowed actions).
    :param airfoil: current state.
    :return future_states: list with all allowed future states.
    """
    future_states = []
    max_chamber = int(airfoil / 1000)
    pos_chamber = int((airfoil - max_chamber * 1000) / 100)
    thickness = airfoil - max_chamber * 1000 - pos_chamber * 100
    future_states.append(airfoil)  # store the same state (as the network can remain on same state)
    # Get all possible new states
    if max_chamber == 1:
        future_states.append(airfoil + 1000)
    elif max_chamber == 9:
        future_states.append(airfoil - 1000)
    else:
        future_states.append(airfoil + 1000)
        future_states.append(airfoil - 1000)
    if pos_chamber == 1:
        future_states.append(airfoil + 100)
    elif pos_chamber == 9:
        future_states.append(airfoil - 100)
    else:
        future_states.append(airfoil + 100)
        future_states.append(airfoil - 100)
    if thickness == 1:
        future_states.append(airfoil + 1)
    elif thickness == 40:
        future_states.append(airfoil - 1)
    else:
        future_states.append(airfoil + 1)
        future_states.append(airfoil - 1)

    return future_states


def NACA_airfoils():
    """
    Compute all feasible airfoils following the guidelines given on the coursework.
    :return airfoils: list with all allowed airfoils NACA 4-digit names.
    """
    airfoils = []
    all = np.arange(1101, 10000, 1)
    for airfoil in all:
        if airfoil % 100 <= 40:  # allowed thickness must be equal or smaller than 40
            if airfoil % 100 != 0:  # allowed thickness must be non-zero
                airfoils.append(airfoil)
    return airfoils

########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                             INM707 - Garcia Plaza, Albert                                            #
########################################################################################################################

import os
import subprocess
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt


class Body():
    """
    Initialise each airfoil as a Python's object with cl and geometry as main attributes
    """
    def __init__(self, airfoil, nPoints=100):
        """
        When is called, initialize the airfoil and perform the simulation to retrieve the lift coefficient.
        :param airfoil: NACA 4-digit name.
        :param nPoints: number of points to discretise the airfoil (more points, more procession but more solving)
        """
        self.cwd = os.getcwd()  # get current path
        self.nPoints = nPoints  # set the number of points to discretise the airfoil
        self.airfoil = airfoil  # set the airfoil NACA 4-digit name
        self.xPoints, self.yPoints = NACA_generator(str(self.airfoil), nPoints)  # generate the discretised points

        clean()  # ensure that the simulation folder is clear and ready to perform a new case simulation

        self.generate_geo()  # generate the geometry file and create the mesh
        self.gmsh2openfoam()  # convert this mesh into a file readable by OpenFOAM
        self.modify_polymesh() # perform the last modification on the OpenFOAM case files just before the simulation
        self.cl = run()  # simulate and get the lift coefficient

    def generate_geo(self):
        """
        Given the airfoil discretised points, generate the C-shaped mesh using GMSH software's API.
        """

        dst = self.cwd + "/airFoil2D/" + str(self.airfoil) + ".geo"  # directory where the geometry file wil be stored
        f = open(dst, 'w+')  # this geometric file is nothing but a plain .txt file with the specific coordinates
        linePointer = 1  # Pointer to store the trailing line of the .txt file.

        # Points writing
        loopSequence = ""
        for i in range(len(self.xPoints)):
            line = "Point(" + str(linePointer) + ") = {" + str(self.xPoints[i]) + ", " + str(self.yPoints[i]) + \
                   ", 0, 0.02};\n"
            f.write(line)
            loopSequence += str(i+1) + ","
            linePointer += 1

        # Create the loop along points
        line = "Spline(" + str(linePointer) + ") = {" + loopSequence[:-1] + ",1};\n"
        f.write(line)
        linePointer += 1
        line = "Line Loop(" + str(linePointer) + ") = {" + str(linePointer-1) + "};\n"
        f.write(line)
        airfoilLoop = linePointer
        linePointer += 1

        # Create the control volume
        line = "Point(" + str(linePointer) + ") = {0, 4, 0, 0.15};\n"
        linePointer += 1
        f.write(line)
        line = "Point(" + str(linePointer) + ") = {0, -4, 0, 0.15};\n"
        linePointer += 1
        f.write(line)
        line = "Point(" + str(linePointer) + ") = {5, -4, 0, 0.15};\n"
        linePointer += 1
        f.write(line)
        line = "Point(" + str(linePointer) + ") = {5, 4, 0, 0.15};\n"
        linePointer += 1
        f.write(line)
        line = "Line(" + str(linePointer) + ") = {" + str(linePointer-1) + "," + str(linePointer-4) + "};\n"
        linePointer += 1
        f.write(line)
        line = "Line(" + str(linePointer) + ") = {" + str(linePointer - 3) + "," + str(linePointer - 2) + "};\n"
        linePointer += 1
        f.write(line)
        line = "Line(" + str(linePointer) + ") = {" + str(linePointer - 5) + "," + str(linePointer - 4) + "};\n"
        linePointer += 1
        f.write(line)
        line = "Point(" + str(linePointer) + ") = {0, 0, 0, 0.02};\n"
        linePointer += 1
        f.write(line)
        line = "Circle(" + str(linePointer) + ") = {" + str(linePointer - 8) + "," + str(linePointer - 1) + "," + \
               str(linePointer - 7) + "};\n"
        linePointer += 1
        f.write(line)
        line = "Line Loop(" + str(linePointer) + ") = {" + str(linePointer - 1) + "," + str(linePointer - 3) + "," + \
               str(linePointer - 4) + "," + str(linePointer - 5) + "};\n"
        controlVolumeLoop = linePointer
        linePointer += 1
        f.write(line)

        # Create surface and extrude it
        line = "Plane Surface(" + str(linePointer) + ") = {" + str(controlVolumeLoop) + "," + str(airfoilLoop) + "};\n"
        f.write(line)
        line = "Recombine Surface{" + str(linePointer) + "};\n"
        f.write(line)
        line = "SurfaceVector[] = Extrude {0, 0, 0.1} {Surface{" + str(linePointer) + "}; Layers{1}; Recombine;};"
        f.write(line)

        f.write("\n")
        f.write("Physical Surface(\"inlet\") = {224};\n")
        f.write("Physical Surface(\"outlet\") = {232};\n")
        f.write("Physical Surface(\"top\") = {236};\n")
        f.write("Physical Surface(\"bottom\") = {228};\n")
        f.write("Physical Surface(\"frontAndBack\") = {214, 241};\n")
        f.write("Physical Surface(\"walls\") = {240};\n")
        f.write("Physical Volume(\"internal\") = {1};\n")

        # Close the file and copy it to the simulation folder renaming it
        f.close()

        # Mesh the file
        cmd = "cd airFoil2D && gmsh " + str(self.airfoil) + ".geo -3"
        subprocess.call(cmd, shell=True)

    def gmsh2openfoam(self):
        """
        Convert the generated mesh file into .msh format, which can be read by OpenFOAM
        """
        cmd = "cd airFoil2D && gmshToFoam " + str(self.airfoil) + ".msh"
        subprocess.call(cmd, shell=True)
        time.sleep(2)

    def modify_polymesh(self):
        """
        Modify OpenFOAM case setup such as boundary conditions, thermodynamical properties, and so on.
        """
        src = self.cwd + "/airFoil2D/constant/polyMesh/boundary"
        with open(src, 'r') as f:
            data = f.readlines()
            data[21] = data[21].replace("patch", "empty")
            data[22] = data[22].replace("patch", "empty")
            data[29] = data[29].replace("patch", "inlet")
            data[36] = data[36].replace("patch", "outlet")
            data[43] = data[43].replace("patch", "outlet")
            data[50] = data[50].replace("patch", "outlet")
            data[56] = data[56].replace("patch", "wall")
            data[57] = data[57].replace("patch", "wall")
        with open(src, 'w') as f:
            f.writelines(data)


def clean():
    """
    Clean the simulation folder from results already simulated.
    """
    subprocess.call("./airFoil2D/Allclean", shell=True)
    time.sleep(2)


def run():
    """
    Run the OpenFOAM simulation through bash commands.
    """
    rootPath = os.getcwd()

    # Execute the simulation until is finished (waiting time of 2 seconds with no changes)
    subprocess.call("./airFoil2D/Allrun", shell=True)

    # One simulation's results output folder is created every time-step until the given final time (or convergence is
    #   reached). So, the code will wait 4 second to check if new folder have been created, if not, the simulation has
    #   finished and the lift coefficient will be retrieved from the folder with bigger name (folder's name represents
    #   simulation time).
    while True:
        files = os.listdir(rootPath + "/airFoil2D/")
        time.sleep(4)  # after many trials, has been proved that any simulation lasts more than 4 seconds
        files_ = os.listdir((rootPath + "/airFoil2D/"))
        if files == files_:  # if after 4 seconds no new folder has been created, simulation has converged
            break

    # Save the coefficient values generated
    path = rootPath + '/airFoil2D/postProcessing/forces/0/forceCoeffs.dat'  # file where OpenFOAM stores the results
    coeffs = open(path, 'r')
    line = coeffs.read().split('\n')[-2]
    values = line.split('\t')
    cl = abs(float(values[2]))
    coeffs.close()

    return cl

# The following two functions have been hacked from the original code developed by https://github.com/dgorissen/naca
def linspace(start, stop, steps):
    """
    Emulate Matlab linspace
    """
    return [start + (stop - start) * i / (steps - 1) for i in range(steps)]


def NACA_generator(number, n):
    """
    Returns 2*n+1 points in [0 1] for the given 4 digit NACA number string
    """
    m = float(number[0]) / 100.0
    p = float(number[1]) / 10.0
    t = float(number[2:]) / 100.0
    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843
    a4 = -0.1015

    beta = linspace(0.0, np.pi, n + 1)
    x = [(0.5 * (1.0 - np.cos(xx))) for xx in beta]  # Half cosine based spacing

    yt = [5 * t * (a0 * np.sqrt(xx) + a1 * xx + a2 * pow(xx, 2) + a3 * pow(xx, 3) + a4 * pow(xx, 4)) for xx in x]
    xc1 = [xx for xx in x if xx <= p]
    xc2 = [xx for xx in x if xx > p]

    if p == 0:
        xu = x
        yu = yt
        xl = x
        yl = [-xx for xx in yt]

    else:
        yc1 = [m / pow(p, 2) * xx * (2 * p - xx) for xx in xc1]
        yc2 = [m / pow(1 - p, 2) * (1 - 2 * p + xx) * (1 - xx) for xx in xc2]
        zc = yc1 + yc2
        dyc1_dx = [m / pow(p, 2) * (2 * p - 2 * xx) for xx in xc1]
        dyc2_dx = [m / pow(1 - p, 2) * (2 * p - 2 * xx) for xx in xc2]
        dyc_dx = dyc1_dx + dyc2_dx
        theta = [np.arctan(xx) for xx in dyc_dx]
        xu = [xx - yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)]
        yu = [xx + yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)]
        xl = [xx + yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)]
        yl = [xx - yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

    X = xu[::-1] + xl[1:]
    Z = yu[::-1] + yl[1:]

    return X, Z


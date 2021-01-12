########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                             INM707 - Garcia Plaza, Albert                                            #
########################################################################################################################
import os
import subprocess
import time

from utils import *


class Body:
    """
    Initialise each airfoil as a Python's object with cl and geometry as main attributes
    """
    def __init__(self, state, name='bodySimulated'):
        """
        When is called, initialize the airfoil and perform the simulation to retrieve the lift coefficient.
        :param name: name of the given body (to store its properties)
        :param control_points: control points controlled by the algorithm as a np array with shape
            [[x1, y1], [x2, y2], ...., [xn, yn]].
        :param nPoints: number of points to discretize the airfoil (more points, more procession but more solving).
        """
        self.cwd = os.getcwd()  # get current path
        self.name = name
        self.xPoints, self.yPoints = create_spline(state)  # generate the discretized body

        clean()  # ensure that the simulation folder is clear and ready to perform a new case simulation

        self.generate_geo()  # generate the geometry file and create the mesh
        self.gmsh2openfoam()  # convert this mesh into a file readable by OpenFOAM
        self.modify_polymesh()  # perform the last modification on the OpenFOAM case files just before the simulation
        self.cl, self.cd = run()  # simulate and get the lift coefficient

    def generate_geo(self):
        """
        Given the airfoil discretised points, generate the C-shaped mesh using GMSH software's API.
        """

        dst = self.cwd + "/airFoil2D/" + str(self.name) + ".geo"  # directory where the geometry file wil be stored
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
        cmd = "cd airFoil2D && gmsh " + str(self.name) + ".geo -3"
        subprocess.call(cmd, shell=True)

    def gmsh2openfoam(self):
        """
        Convert the generated mesh file into .msh format, which can be read by OpenFOAM
        """
        cmd = "cd airFoil2D && gmshToFoam " + str(self.name) + ".msh"
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
    cd = abs(float(values[1]))
    cl = abs(float(values[2]))
    coeffs.close()

    return cl, cd

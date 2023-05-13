"""
Reads a series of cube files, and prepares a numpy array of movie
"""

import numpy as np
from os.path import isfile
from sys import exit
from scipy.constants import physical_constants


class CubeFiles:
    """
    Cube Class:
    Includes a bunch of methods to manipulate cube data
    """

    def __init__(self):
        self.natoms = None
        self.comment1 = ''
        self.comment2 = ''
        self.origin = None
        self.NX = None
        self.NY = None
        self.NZ = None
        self.X = None
        self.Y = None
        self.Z = None
        self.atoms = None
        self.atomsXYZ = None
        self.data = None
        self.no_cubes_read = int(0)
        self.bohrA = physical_constants['Bohr radius'][0] * 1e10

    def errore(self, no=-1):
        if no == 1:
            raise Exception("Only densities in the cube files are allowed to change")
        else:
            raise Exception("An exception has occured")

    def read_cube(self, fname):
        """
        Method to read a single cube file. Needs a filename. Needs an input filename,Appends the density data.
        """

        with open(fname, 'r') as fin:
            print("Reading : ", fname, " as cube number",self.no_cubes_read+1)
            self.comment1 = fin.readline()  # Save 1st comment
            self.comment2 = fin.readline()  # Save 2nd comment
            nOrigin = fin.readline().split()  # Number of Atoms and Origin
            self.natoms = int(nOrigin[0])  # Number of Atoms
            self.origin = np.array([float(nOrigin[1]), float(nOrigin[2]), float(nOrigin[3])])  # Position of Origin
            nVoxel = fin.readline().split()  # Number of Voxels
            self.NX = int(nVoxel[0])
            self.X = np.array([float(nVoxel[1]), float(nVoxel[2]), float(nVoxel[3])])
            nVoxel = fin.readline().split()  #
            self.NY = int(nVoxel[0])
            self.Y = np.array([float(nVoxel[1]), float(nVoxel[2]), float(nVoxel[3])])
            nVoxel = fin.readline().split()  #
            self.NZ = int(nVoxel[0])
            self.Z = np.array([float(nVoxel[1]), float(nVoxel[2]), float(nVoxel[3])])
            self.atoms = []
            self.atomsXYZ = []
            for atom in range(self.natoms):
                line = fin.readline().split()
                self.atoms.append(line[0])
                self.atomsXYZ.append(list(map(float, [line[2], line[3], line[4]])))
            if self.no_cubes_read == 0 :
                self.data = np.zeros((self.NX, self.NY, self.NZ , 1000))
            i = int(0)
            for s in fin:
                for v in s.split():
                    self.data[int(i / (self.NY * self.NZ)), int((i / self.NZ) % self.NY), int(
                        i % self.NZ), self.no_cubes_read] = float(v)
                    i += 1
            # if i != self.NX*self.NY*self.NZ: raise NameError, "FSCK!"
            self.no_cubes_read += 1
        return

    def density_as_numpy_3d(self):
        """
        Return the read density data as a series of numpy arrays
        :return:
        """
        return self.data

    def get_voxel_coordinate_angstrom(self, i, j, k):
        """
        Get the coordinate of voxel in angstroms
        :param i: Voxel identified i
        :param j: Voxel identified j
        :param k: Voxel identified k
        :return: x,y,z
        """
        x = self.bohrA * (i * self.X[0] + j * self.Y[0] + k * self.Z[0])
        y = self.bohrA * (i * self.X[1] + j * self.Y[1] + k * self.Z[1])
        z = self.bohrA * (i * self.X[2] + j * self.Y[2] + k * self.Z[2])
        return x, y, z


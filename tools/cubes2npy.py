"""
Reads a series of cube files, and prepares a numpy array of movie
"""
import numpy
import numpy as np
from os.path import isfile
from sys import exit
from scipy.constants import physical_constants


class CubeFiles:
    """
    Cube Class:
    Includes a bunch of methods to read and manipulate electron densities
    written as gaussian cube format
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
        self.data2d = None
        self.no_cubes_read = int(0)
        self.bohrA = physical_constants['Bohr radius'][0] * 1e10
        self.max_cubes=1000
        self.i_min=0
        self.j_min=0
        self.k_min=0

    def errore(self, no=-1):
        if no == 1:
            raise Exception("Only densities in the cube files are allowed to change")
        else:
            raise Exception("An exception has occured")

    def read_cube(self, fname):
        """
        Method to read a single cube file. Needs a filename. Needs an input filename,Appends the density data.
        :param fname: Name of the cube file
        :return:
        """

        with open(fname, 'r') as fin:
            print("Reading : ", fname, " as cube number", self.no_cubes_read + 1)
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
            if self.no_cubes_read == 0:
                self.data = np.array(np.zeros((self.NX, self.NY, self.NZ,1)))
            read_data=np.array(np.zeros((self.NX, self.NY, self.NZ,1)))
            i = int(0)
            for s in fin:
                for v in s.split():
                    #The data is intended for Tensorflow, so "batch size"=time comes first
                    read_data[int(i / (self.NY * self.NZ)), int((i / self.NZ) % self.NY), int(
                        i % self.NZ),0] = float(v)
                    i += 1
            # if i != self.NX*self.NY*self.NZ: raise NameError, "FSCK!"
            if self.no_cubes_read == 0:
                self.data=np.copy(read_data)
            else:
                self.data=np.concatenate((self.data,read_data),axis=-1)
            self.no_cubes_read += 1
            if self.no_cubes_read > self.max_cubes :
                exit()
        return

    def read_cube_2Dslice(self, fname, slice_normal, slice_position):
        """
        Reads a 2D slice from a cube file and appends to data_2D
        :param fname:
        :param slice_normal: 'x', 'y' or 'z'
        :param slice_position: integer between 50-100 meaning the percentile of N>
        :return:
        """

        with open(fname, 'r') as fin:
            print("Reading a 2D slice (norm: ", slice_normal, ",pos:%", slice_position, ") from", fname,
                  " as cube number", self.no_cubes_read + 1)
            fin.readline()  # Save 1st comment
            fin.readline()  # Save 2nd comment
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
            if self.no_cubes_read == 0:
                if slice_normal == 'x':
                    self.data2d = np.zeros((self.NY, self.NZ, 10000))
                if slice_normal == 'y':
                    self.data2d = np.zeros((self.NZ, self.NX, 10000))
                if slice_normal == 'z':
                    self.data2d = np.zeros((self.NX, self.NY, 10000))
            data = np.zeros((self.NX, self.NY, self.NZ))
            i = int(0)
            for s in fin:
                for v in s.split():
                    data[int(i / (self.NY * self.NZ)), int((i / self.NZ) % self.NY), int(
                        i % self.NZ)] = float(v)
                    i += 1
            # if i != self.NX*self.NY*self.NZ: raise NameError, "FSCK!"
            #The data is intended for Tensorflow, so "batch size"=time comes first
            if slice_normal == 'x':
                slice_index = int(np.floor(self.NX * slice_position / 100.0))
                self.data2d[self.no_cubes_read,:, :] = data[slice_index, :, :]
            if slice_normal == 'y':
                slice_index = int(np.floor(self.NY * slice_position / 100.0))
                self.data2d[self.no_cubes_read, :, :] = data[:, slice_index, :]
            if slice_normal == 'z':
                slice_index = int(np.floor(self.NZ * slice_position / 100.0))
                self.data2d[self.no_cubes_read, :, :] = data[:, :, slice_index]
            self.no_cubes_read += 1
        return

    def density_as_numpy(self):
        """
        Return the read density data as a series of numpy arrays
        :return:
        """
        return self.data

    def density_as_numpy_2d(self):
        """
        Returns the 2D reduced density from data file
        :return:
        """
        return self.data2d

    def get_voxel_coordinate_angstrom(self, i, j, k):
        """
        Get the coordinate of voxel in angstroms
        :param i: Voxel identified i
        :param j: Voxel identified j
        :param k: Voxel identified k
        :return: x,y,z
        """
        x = self.bohrA * ((i+self.i_min) * self.X[0] + (j+self.j_min) * self.Y[0] + (k+self.k_min) * self.Z[0])
        y = self.bohrA * ((i+self.i_min) * self.X[1] + (j+self.j_min) * self.Y[1] + (k+self.k_min) * self.Z[1])
        z = self.bohrA * ((i+self.i_min) * self.X[2] + (j+self.j_min) * self.Y[2] + (k+self.k_min) * self.Z[2])
        return x, y, z

    def trim_cube_data(self,i_min,i_max,j_min,j_max,k_min,k_max):
        """
        Trims the cube time series to save memory. The coordinate queries are updated accordingly
        :param i_min: minimum i index
        :param i_max: maximum i index
        :param j_min: minimum j index
        :param j_max: maximum j index
        :param k_min: minimum k index
        :param k_max: maximum k index
        :return:
        """
        self.i_min=i_min
        self.j_min=j_min
        self.k_min=k_min
        data=np.zeros(((i_max-i_min),(j_max-j_min),(k_max-k_min),self.no_cubes_read))
        data=np.copy(self.data[i_min:i_max,j_min:j_max,k_min:k_max,:])
        self.data=None
        self.data=np.copy(data)
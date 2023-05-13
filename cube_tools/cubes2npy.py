"""
Reads a series of cube files, and prepares a numpy array of movie
the cube class is originally from funkymunkycool / Cube-Toolz
"""

import numpy as np
from os.path import isfile
from sys import exit, argv
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.constants import physical_constants
from skimage import transform
from scipy.spatial.transform import Rotation as R


class Cube:
    """
    Cube Class:
    Includes a bunch of methods to manipulate cube data
    """

    def __init__(self, fname=None):
        self.filename = None
        self.natoms = 0
        self.comment1 = 0
        self.comment2 = 0
        self.origin = np.array([0, 0, 0])
        self.NX = 0
        self.NY = 0
        self.NZ = 0
        self.X = 0
        self.Y = 0
        self.Z = 0
        self.atoms = ['0']
        self.atomsXYZ = [0, 0, 0]
        self.data = [0]
        if fname is not None:
            try:
                self.read_cube(fname)
            except IOError as e:
                print("File used as input: %s" % fname)
                print("File error ({0}): {1}".format(e.errno, e.strerror))
                exit()
        return None

    def read_cube(self, fname):
        """
        Method to read cube file. Just needs the filename
        """

        with open(fname, 'r') as fin:
            self.filename = fname
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
            self.data = np.zeros((self.NX, self.NY, self.NZ))
            i = int(0)
            for s in fin:
                for v in s.split():
                    self.data[int(i / (self.NY * self.NZ)), int((i / self.NZ) % self.NY), int(i % self.NZ)] = float(v)
                    i += 1
            # if i != self.NX*self.NY*self.NZ: raise NameError, "FSCK!"
        return None

    def write_cube(self, fname, comment=''):
        """
        Write out a Gaussian Cube file
        """
        try:
            with open(fname, 'w') as fout:
                if len(comment.split('\n')) < 2:
                    fout.write('\n')
                fout.write('%s\n' % comment)
                fout.write("%4d %.6f %.6f %.6f\n" % (self.natoms, self.origin[0], self.origin[1], self.origin[2]))
                fout.write("%4d %.6f %.6f %.6f\n" % (self.NX, self.X[0], self.X[1], self.X[2]))
                fout.write("%4d %.6f %.6f %.6f\n" % (self.NY, self.Y[0], self.Y[1], self.Y[2]))
                fout.write("%4d %.6f %.6f %.6f\n" % (self.NZ, self.Z[0], self.Z[1], self.Z[2]))
                for atom, xyz in zip(self.atoms, self.atomsXYZ):
                    fout.write("%s %d %6.3f %6.3f %6.3f\n" % (atom, 0, xyz[0], xyz[1], xyz[2]))
                for ix in range(self.NX):
                    for iy in range(self.NY):
                        for iz in range(self.NZ):
                            fout.write("%.5e " % self.data[ix, iy, iz]),
                            if (iz % 6 == 5): fout.write('\n')
                        fout.write("\n")
        except IOError as e:
            print("File used as output does not work: %s" % fname)
            print("File error ({0}): {1}".format(e.errno, e.strerror))
            exit()
        return None

    def square_cube(self, power=2):

        """
        Function to raise cube data to a power. Squares cube data by default.
        """
        self.data = self.data ** power
        print(power)
        return None

    def rotate_cube(self, angle, axes=None):
        """
        Rotate cube data around a plane. The plane is defined in the axes variable with origin set as point. For example, to rotate along the xy plane, axes would be defined as (0,1).
        """
        if 0 not in axes:
            rotAxis = 'x'
        elif 1 not in axes:
            rotAxis = 'y'
        elif 2 not in axes:
            rotAxis = 'z'

        # Rotate the atoms
        r = R.from_euler(rotAxis, angle, degrees=True)
        self.atomsXYZ = r.apply(self.atomsXYZ)

        # Centre of new cell
        centreNewCell = np.sum(r.apply([self.X * self.NX, self.Y * self.NY, self.Z * self.NZ]) / 2., axis=0)

        # Rotate the cube data
        self.data = ndimage.rotate(self.data, angle, axes=axes, mode='wrap')
        self.NX, self.NY, self.NZ = np.shape(self.data)

        # Move atoms' centre of mass to centre of cell
        newCentre = ((self.X * self.NX + self.Y * self.NY + self.Z * self.NZ)) / 2.
        centreDiff = centreNewCell - newCentre

        self.atomsXYZ = self.atomsXYZ - centreDiff

        # Make sure atoms are in cell
        self.atomsXYZ = self.atomsXYZ % (self.X * self.NX + self.Y * self.NY + self.Z * self.NZ)

        return None

    def translate_cube(self, tVector):
        '''
        Translate cube data by some vector. The vector is given as a list to the tVector function.
        '''
        self.data = ndimage.shift(self.data, tVector, mode='wrap')
        return None

    def planar_average(self, axis):
        '''
        Calculate the planar average along an axis. The axis is given as a string of either x,y or z.
        '''
        bohrA = physical_constants['Bohr radius'][0] * 1e10
        vol = np.linalg.det(np.array([self.X, self.Y, self.Z]))
        dx, dy, dz = (self.X + self.Y + self.Z) * bohrA

        if axis == 'x':
            PlanAv = np.array(
                [[nx * self.X[0] * bohrA, (np.sum(self.data[nx, ::]) * vol) / dx] for nx in range(self.NX)])
        elif axis == 'y':
            PlanAv = np.array(
                [[ny * self.Y[1] * bohrA, (np.sum(self.data[:, ny, :]) * vol) / dy] for ny in range(self.NY)])
        elif axis == 'z':
            PlanAv = np.array(
                [[nz * self.Z[2] * bohrA, np.sum(self.data[:, :, nz] * vol) / dz] for nz in range(self.NZ)])
        else:
            print('%s' % 'No axis specified! Planar average will return zero and fail.')
            PlanAv = 0.0
        return PlanAv

    def planar_averageG(self, axis, sigma):
        '''
        Broaden the planar average along an axis. The axis is given as a string of either x,y or z. A broadening value is also needed.
        '''
        PlanAvG = self.planar_average(axis)
        PlanAvG[:, 1] = gaussian_filter(PlanAvG[:, 1], sigma)
        return PlanAvG

    def cube_int(self):
        '''
        Integrate the entire cube data.
        '''
        angstrom = physical_constants['Bohr radius'][0] * 1e10
        vol = np.linalg.det(np.array([self.X, self.Y, self.Z]))
        edensity = np.sum(self.data)
        nelectron = vol * edensity

        # print 'Number of electrons: %.7g' % (nelectron)
        return nelectron

    def cube_int_atom(self, atomID, radius):
        '''
        Integrate the cube data in a sphere around a particular atom. Needs the atom number (note that atom 0 is the first atom). Also needs a radius of the sphere.
        '''
        voxelMatrix = np.array([self.X, self.Y, self.Z])
        radius *= 1 / (physical_constants['Bohr radius'][0] * 1e10)
        vol = np.linalg.det(voxelMatrix)
        atomXYZ = np.array(self.atomsXYZ[atomID])  # + self.origin

        ZYX = np.ogrid[:self.NX, :self.NY, :self.NZ]
        dZYX = self.Z + self.Y + self.X
        ZYX = ZYX * dZYX

        dist_from_center = np.linalg.norm(ZYX - atomXYZ)
        mask = dist_from_center <= radius

        nelectron = np.sum(mask * self.data * vol)

        return nelectron

    def cube_int_ref(self, ref, radius):
        '''
        Integrate the cube data in a sphere around a point. Also needs the point as a list.
        '''
        voxelMatrix = [self.X, self.Y, self.Z]
        vol = np.linalg.det(voxelMatrix)
        ref = np.array(ref) * (1 / (physical_constants['Bohr radius'][0] * 1e10))
        radius *= 1 / (physical_constants['Bohr radius'][0] * 1e10)

        ZYX = np.array(np.ogrid[:self.NX, :self.NY, :self.NZ], dtype=object)
        dZYX = self.Z + self.Y + self.X
        ZYX *= dZYX

        dist_from_center = np.linalg.norm(ZYX - ref)
        mask = dist_from_center <= radius

        nelectron = np.sum(mask * self.data * vol)

        return nelectron

    def super_cube(self, new_size):
        '''
        Function to make a new cube supercell. Takes in 3D list of how big the supercell should be.
        '''
        cell = np.array((self.NX * self.X, self.NY * self.Y, self.Z * self.NZ))
        new_data = np.zeros([new_size[0] * self.NX, new_size[1] * self.NY, new_size[2] * self.NZ])
        new_xyz = self.atomsXYZ
        n_newcells = np.prod(new_size)
        new_xyz = np.array(tuple(self.atomsXYZ) * n_newcells)
        counter = 0
        for x in range(new_size[0]):
            for y in range(new_size[1]):
                for z in range(new_size[2]):
                    new_data[x * self.NX:(x + 1) * self.NX, y * self.NY:(y + 1) * self.NY,
                    z * self.NZ:(z + 1) * self.NZ] += self.data
                    new_xyz[counter * self.natoms:(counter + 1) * self.natoms] = self.atomsXYZ + (
                            (np.array(cell[0]) * x) + (np.array(cell[1]) * y) + (np.array(cell[2]) * z))
                    counter += 1
        new_data = transform.rescale(new_data, 1 / np.mean(new_size), order=3)
        self.data = new_data
        self.X = ((self.NX * self.X) / float(np.shape(new_data)[0])) * new_size[0]
        self.Y = ((self.NY * self.Y) / float(np.shape(new_data)[1])) * new_size[1]
        self.Z = ((self.NZ * self.Z) / float(np.shape(new_data)[2])) * new_size[2]
        self.NX, self.NY, self.NZ = np.shape(new_data)
        self.atomsXYZ = new_xyz
        self.atoms *= len(self.atomsXYZ)
        self.natoms = len(new_xyz)
        return None

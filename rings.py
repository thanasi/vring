#!/usr/bin/env python
from __future__ import division
import numpy as np
# from mayavi import mlab

from progress.bar import Bar

class Curve(object):

    def __init__(self, x, y, z):

        if len(x) != len(y):
            raise AttributeError("x and y must contain the same number of points")
        if len(x) != len(z):
            raise AttributeError("x and z must contain the same number of points")

        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

        self.s = np.array(zip(x,y,z))

        self.N = self.x.shape[0]

        self.curvature = np.zeros_like(x)
        self.normal = np.zeros_like(self.s)
        self.nx = np.zeros_like(x)
        self.ny = np.zeros_like(x)
        self.nz = np.zeros_like(x)
        self.tangent = np.zeros_like(self.s)
        self.tx = np.zeros_like(x)
        self.ty = np.zeros_like(x)
        self.tz = np.zeros_like(x)

        self.update_geometric_properties()

    # TODO: Something's still wrong with curvature calculation
    def update_geometric_properties(self):
        # derivatives are central differences (second order) w/ linear interpolation
        ds = (np.roll(self.s, -1, axis=0) - np.roll(self.s, 1, axis=0)) / 2
        d2s = np.roll(ds, -1, axis=0) - np.roll(ds, 1, axis=0) / 2

        self.tangent = ds / Norm2(ds)[:, np.newaxis]

        self.curvature = Norm2(np.cross(ds, d2s, axis=1)) / Norm2(ds)**3

        self.normal = d2s / Norm2(d2s)[:,np.newaxis]

        self.nx, self.ny, self.nz = zip(*self.normal)
        self.tx, self.ty, self.tx = zip(*self.tangent)


    def create_vortex_tube(self, w0, vorticity_type='tangent'):

        self.update_geometric_properties()

        if vorticity_type=='tangent':
            return VortexTube(self.x, self.y, self.z, w0*self.tx, w0*self.ty, w0*self.tz)

        else:
            return VortexTube(self.x, self.y, self.z,
                              np.zeros_like(self.tx),
                              np.zeros_like(self.ty),
                              np.zeros_like(self.tz))

class VortexTube(Curve):
    def __init__(self, x, y, z, wx, wy, wz):
        super(VortexTube, self).__init__(x, y, z)

        # set up vorticity field
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.w = np.array(zip(wx, wy, wz))

        # initialize velocity field
        self.ux = np.zeros_like(wx)
        self.uy = np.zeros_like(wx)
        self.uz = np.zeros_like(wx)

        self.u = np.zeros_like(self.w)

        self.impulse = np.zeros(3, dtype = np.float)
        self.energy = 0

        self.calc_biotsavart_vel()
        self.calc_impulse()
        self.calc_energy()


    def calc_biotsavart_vel(self):
        # get the velocity field at every point
        for i in range(self.N):

            mask = (np.arange(self.N)==i)

            # vector difference of current position with all other vortex positions
            xdiff = np.repeat(self.s[i][np.newaxis, :], self.N, axis=0) - self.s

            # norm of xdiff
            nxdiff = Norm2(xdiff)
            nxdiff[mask] = 1

            integrand = np.cross(self.w, xdiff, axis=1) / (nxdiff**3)[:,np.newaxis]
            integrand[mask] = np.zeros(3)

            self.u[i] = integrand.sum(0) / (4*np.pi)
            self.ux, self.uy, self.uz = zip(*self.u)

    # TODO: Need to validate these calculations
    def calc_energy(self):
        # scalar
        self.energy = -(self.u * np.cross(self.s, self.w, axis=1)).sum()

    def calc_impulse(self):
        # vector
        self.impulse = 0.5 * np.cross(self.s, self.w, axis=1).sum(0)


def Circle(radius, center=(0,0,0), num_points=1000):
    ''' initialize a circular path '''

    theta = np.arange(0,2*np.pi, 2*np.pi/num_points)

    cx = radius * np.cos(theta) + center[0]
    cy = radius * np.sin(theta) + center[1]
    cz = center[2] * np.ones_like(theta)

    return Curve(cx,cy,cz)

def Bridged_Circles(radius, offset=0.01, center=(0,0,0), num_points=1000):

    theta = np.arange(0, 2*np.pi, 2*np.pi/num_points)
    r = radius * (np.cos(theta)**2) + offset

    cx = r * np.cos(theta) + center[0]
    cy = r * np.sin(theta) + center[1]
    cz = center[2] * np.ones_like(theta)


    return Curve(cx, cy, cz)

def Norm2(vector_list):
    # R^n -> R^(n-1)
    # assumes a list of points and takes the L2 norm of each point
    return np.sqrt((vector_list**2).sum(1))


if __name__ == "__main__":
    pass
import numpy as np


class Grid1D:
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step
        self.span = np.arange(start, end+step, step)
        self.length = len(self.span)
        self.range = range(self.length)

    def index_of(self, val):
        return np.where(self.span == val)[0][0]

    def closest_index(self, val):
        return np.argmin(np.abs(self.span - val))

    def __str__(self):
        return str({"start": self.start, "end": self.end, "step": self.step})


class SpaceGrid2D:
    def __init__(self, x_start, x_end, dx, z_start, z_end, dz=1):
        self.x_start = x_start
        self.x_end = x_end
        self.dx = dx
        self.xgrid = Grid1D(x_start, x_end, dx)
        self.x_range = self.xgrid.range
        self.Nx = self.xgrid.length
        self.x = self.xgrid.span

        self.z_start = z_start
        self.z_end = z_end
        self.dz = dz
        self.zgrid = Grid1D(z_start, z_end, dz)
        self.z_range = self.zgrid.range
        self.Nz = self.zgrid.length
        self.z = self.zgrid.span

    def index_of_x(self, x_nm):
        return np.where(self.x == x_nm)[0][0]

    def xrange(self, start, end, include_end=False):
        """
        Returns a vector of value in the given range matching with the grid dx spacing
        :param start:
        :param end:
        :param include_end:
        :return:
        """
        if (len(np.where(self.x == start)[0]) == 0) or (len(np.where(self.x == end)[0]) == 0):
            raise Exception("The range requested is not aligned with the grid")

        if include_end:
            return np.arange(start, end+self.dx, self.dx)
        else:
            return np.arange(start, end, self.dx)


class SimulationGrid:
    def __init__(self, x_start, x_end, dx, z_start, z_end, dz, t_start, t_end, dt):
        self.x_start = x_start
        self.x_end = x_end
        self.dx = dx
        self.xgrid = Grid1D(x_start, x_end, dx)
        self.x_range = self.xgrid.range
        self.Nx = self.xgrid.length
        self.x = self.xgrid.span

        self.z_start = z_start
        self.z_end = z_end
        self.dz = dz
        self.zgrid = Grid1D(z_start, z_end, dz)
        self.z_range = self.zgrid.range
        self.Nz = self.zgrid.length
        self.z = self.zgrid.span

        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.tgrid = Grid1D(t_start, t_end, dt)
        self.t_range = self.tgrid.range
        self.Nt = self.tgrid.length
        self.t = self.tgrid.span

    def index_of_x(self, x_nm):
        return np.where(self.x == x_nm)[0][0]

    def index_of_t(self, t_ps):
        return np.where(self.t == t_ps)[0][0]

    def xrange(self, start, end, include_end=False):
        """
        Returns a vector of value in the given range matching with the grid dx spacing
        :param start:
        :param end:
        :param include_end:
        :return:
        """
        if (len(np.where(self.x == start)[0]) == 0) or (len(np.where(self.x == end)[0]) == 0):
            raise Exception("The range requested is not aligned with the grid")

        if include_end:
            return np.arange(start, end+self.dx, self.dx)
        else:
            return np.arange(start, end, self.dx)

    @staticmethod
    def __step_to_number_points(start, end, step):
        return round((end-start) / step + 1)

    @staticmethod
    def __create_vector(start, end, n_points):
        return np.linspace(start, end, n_points)

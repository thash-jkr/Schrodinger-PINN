import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

from physics.utils.misc import index_of_nearest, find_resonance
from physics.utils.grids import Grid1D, SpaceGrid2D
from physics.utils.plots import input_output_plot


class COMSOLFrequencySimulation:
    def __init__(self, path, name=None, columns=None):
        self.path = path
        if name is None:
            self.name = self.path
        else:
            self.name = name
            path = self.path + self.name

        default_columns = (columns is None)
        if default_columns:
            columns = ["H", "f", "s21"]

        self.raw_df = pd.read_csv(path, delimiter=r'\s+', header=None, skiprows=5, names=columns)
        if default_columns:
            self.df = self.raw_df.pivot(index="f", columns="H", values="s21")
        self.H0 = self.df.columns
        self.f = self.df.index


class EmptyCavityMeasurement:
    def __init__(self, name, folder, load_raw_data=False):
        """
        A data wrapper for a measurement of an empty cavity, i.e. a cavity without magnetic field, as measured by
        Guillaume's program at IMT.
        :param name: name of the measurement
        :param folder: path to the measurement folder
        :param load_raw_data: load directly from text files created by the measurement software. Else, will load a
        pickle file with reduced size.
        """
        self.folder, self.name = folder, name

        if load_raw_data:
            self.f = np.genfromtxt(folder + name + "\\f_values.txt", delimiter='\n', skip_header=1)

            S_param_delimiter = r','
            for s_ij in ["S11", "S21", "S12", "S22"]:
                self.__dict__[s_ij] = np.genfromtxt(folder + name + "/S/%s/Magnitude.txt" % s_ij,
                                                    delimiter=S_param_delimiter, skip_header=1).transpose()
                self.__dict__[s_ij+"_phase"] = np.genfromtxt(folder + name + "/S/%s/Phase.txt" % s_ij,
                                                             delimiter=S_param_delimiter, skip_header=1).transpose()
        else:
            with open(folder + self.name + ".pkl", "rb") as file:
                class_attributes = pickle.load(file)
            self.f = class_attributes["frequency_GHz"]
            for s_ij in ["S11", "S21", "S12", "S22"]:
                self.__dict__[s_ij] = class_attributes[s_ij]
                self.__dict__[s_ij+"_phase"] = class_attributes[s_ij+"_phase"]

    def pickle(self):
        """
        Dump the object to a lightweight pickle file for faster loading.
        :return:
        """
        # Note that this must be called from outside this script to work
        # See https://www.pythonanywhere.com/forums/topic/27818/#id_post_81907
        fields = {
            "frequency_GHz": self.f,
        }
        for s_ij in ["S11", "S21", "S12", "S22"]:
            fields[s_ij] = self.__dict__[s_ij]
            fields[s_ij+"_phase"] = self.__dict__[s_ij+"_phase"]
        with open(self.folder + self.name + ".pkl", "wb") as file:
            pickle.dump(fields, file)

    def trim_frequency_range_S21(self, f_lims):
        f_min_idx, f_max_idx = index_of_nearest(self.f, f_lims[0]), index_of_nearest(self.f, f_lims[1])
        return self.f[f_min_idx:f_max_idx+1], self.S21[f_min_idx:f_max_idx+1]


class CavityMeasurement:
    def __init__(self, path, name, load_raw_data=True):
        """
        A data wrapper for a measurement of the S param using Vincent's 40GHz VNA.
        :param name: name of the measurement
        :param load_raw_data: load directly from text files created by the measurement software
        """
        self.path = path
        self.name = name
        if load_raw_data:
            self.f = np.genfromtxt(path + name + "/f_values.txt", delimiter='\n', skip_header=1)
            self.H0 = 1e3*np.genfromtxt(path + name + "/H_values.txt", delimiter='\n', skip_header=1)
            self.H0 = np.abs(self.H0)

            S_param_delimiter = r','
            for s_ij in ["S11", "S21", "S12", "S22"]:
                self.__dict__[s_ij] = np.genfromtxt(path + name + "/S/%s/Magnitude.txt" % s_ij,
                                                    delimiter=S_param_delimiter, skip_header=1).transpose()
                self.__dict__[s_ij+"_phase"] = np.genfromtxt(path + name + "/S/%s/Phase.txt" % s_ij,
                                                             delimiter=S_param_delimiter, skip_header=1).transpose()
        else:
            with open(path + name + ".pkl", "rb") as file:
                class_attributes = pickle.load(file)
            self.f = class_attributes["frequency_GHz"]
            self.H0 = class_attributes["static_magnetic_field_mT"]
            for s_ij in ["S11", "S21", "S12", "S22"]:
                self.__dict__[s_ij] = class_attributes[s_ij]
                self.__dict__[s_ij+"_phase"] = class_attributes[s_ij+"_phase"]

    def pickle(self):
        """
        Dump the object to a lightweight pickle file for faster loading.
        :return:
        """
        # Note that this must be called from outside this script to work
        # See https://www.pythonanywhere.com/forums/topic/27818/#id_post_81907
        fields = {
            "frequency_GHz": self.f,
            "static_magnetic_field_mT": self.H0
        }
        for s_ij in ["S11", "S21", "S12", "S22"]:
            fields[s_ij] = self.__dict__[s_ij]
            fields[s_ij+"_phase"] = self.__dict__[s_ij+"_phase"]
        with open(self.path + self.name + ".pkl", "wb") as file:
            pickle.dump(fields, file)

    def display_S_param(self, s_param_label="S21", show=True, H_lim=None, freq_lim=None, colormap=cm.oslo, fig=None, ax=None):
        """
        Display the data in the specified range.
        :return: fig, ax
        """
        H0, f, S_param = self.H0, self.f, self.__dict__[s_param_label]

        if H_lim is not None:
            indices = index_of_nearest(H0, H_lim[0]), index_of_nearest(H0, H_lim[1])
            H0 = H0[indices[0]:indices[1]+1]
            S_param = S_param[:, indices[0]:indices[1]+1]

        if freq_lim is not None:
            indices = index_of_nearest(f, freq_lim[0]), index_of_nearest(f, freq_lim[1])
            f = f[indices[0]:indices[1]+1]
            S_param = S_param[indices[0]:indices[1]+1, :]

        fig, ax = input_output_plot(H0, f, S_param, fig=fig, ax=ax, dB=True, dB_step=20, colorbar_format='%.0f', colorbar_label="|%s| (dB)"%s_param_label)
        ax.set_xlabel("Applied magnetic field (mT)")
        fig.suptitle(self.name)
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def trim_frequency_range(self, f_lims, s_param_label):
        f_min_idx, f_max_idx = index_of_nearest(self.f, f_lims[0]), index_of_nearest(self.f, f_lims[1])
        return self.f[f_min_idx:f_max_idx+1], self.__dict__[s_param_label][f_min_idx:f_max_idx+1, :]

    def plot_slice(self, H_applied, f_lims=None):
        fig, ax = plt.subplots(1, 1)

        if f_lims is None:
            ax.plot(self.f, self.S21[:, index_of_nearest(self.H0, H_applied)])
        else:
            f_min_idx, f_max_idx = index_of_nearest(self.f, f_lims[0]), index_of_nearest(self.f, f_lims[1])
            ax.plot(self.f[f_min_idx:f_max_idx+1], self.S21[f_min_idx:f_max_idx+1, index_of_nearest(self.H0, H_applied)])

            res = find_resonance(self.f[f_min_idx:f_max_idx+1]/1e9, self.S21[f_min_idx:f_max_idx+1, index_of_nearest(self.H0, H_applied)])
            print(res.frequency_GHz)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("$|S_{21}|$ (dB)")
        fig.suptitle("Slice at $H=%0.3f$ mT" % H_applied)
        plt.show()


class FileManager:
    def __init__(self, N_states, grid: SpaceGrid2D, time_grid: Grid1D, geometry, data_folder="../data/"):
        self.grid_str = self._get_grid_str(grid)
        self.time_grid_str = self._get_time_grid_str(time_grid)
        self.geometry_str = self._get_geometry_str(geometry)
        self.data_folder = "../data/"

        self.path = self.data_folder + self.grid_str + "-" + self.time_grid_str + "/"
        self.filename = "N-%d-%s" % (N_states, self.geometry_str)

    def _get_grid_str(self, grid):
        return "xstart-%d-xend-%d-dx-%.1f-Nz-%d" % (abs(grid.x_start), grid.x_end, grid.dx, grid.z_end)

    def _get_time_grid_str(self, time_grid):
        return "tend-%0.1f-dt-%.2f" % (time_grid.end, time_grid.step)

    def _get_geometry_str(self, geometry):
        return "profile"

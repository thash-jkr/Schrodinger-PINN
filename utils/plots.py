import matplotlib.pyplot as plt
from cmcrameri import cm
import numpy as np
from matplotlib.animation import FuncAnimation
from physics.utils.grids import Grid1D


LATEX_TEXT_WIDTH = 510
LATEX_COLUMN_WIDTH = 246
THESIS_TEXT_WIDTH = 0
THESIS_COLUMN_WIDTH = 0
BEAMER_TEXT_WIDTH = 2*307

# input_output_cmap = cm.imola    # Yellow max, green, blue min
# input_output_cmap = cm.lajolla_r  # Yellow max, red, black min.
input_output_cmap = cm.oslo  # White max, blue, black min


def get_figsize(width: float, fraction: float = 1):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    :param width: document textwidth or columnwidth in pts.
    :param fraction: fraction of the width which you wish the figure to occupy, optional.
    :return: tuple, dimension of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def default_plots():
    """
    Defines the default plot params. This is automatically executed when this module is imported.
    :return: None
    """
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['axes.grid'] = False

    # https://matplotlib.org/3.5.0/tutorials/intermediate/autoscale.html
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0


def paper_plots(column_size=LATEX_COLUMN_WIDTH, square=False):
    """
    :param column_size: column size of the paper
    :return:
    """
    plt.rcParams['figure.dpi'] = 400
    figsize = get_figsize(column_size)
    print(figsize)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    if square:
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[0]]
    else:
        plt.rcParams['figure.figsize'] = get_figsize(column_size)

    return plt.rcParams['figure.figsize']


def presentation_plots(DEFAULT_PLOTS, PRESENTATION_PLOTS):
    DEFAULT_PLOTS, PRESENTATION_PLOTS = False, True
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = get_figsize(BEAMER_TEXT_WIDTH)
    plt.rcParams['font.size'] = 13

    return DEFAULT_PLOTS, PRESENTATION_PLOTS


def input_output_spectrum_plot(system, matter_freqs, probe_freqs, detuning_probe_matter_freqs=1e-4, dB=False,
                               title=None, analytical_spectrum=False, show=False):
    matter_freqs = matter_freqs + detuning_probe_matter_freqs
    S21 = system.transmission(probe_freqs, matter_freqs)
    if not dB:
        fig, ax = input_output_plot(matter_freqs-detuning_probe_matter_freqs, probe_freqs, S21)
    else:
        fig, ax = input_output_plot(matter_freqs-detuning_probe_matter_freqs, probe_freqs, 10*np.log10(S21), dB=dB)

    if analytical_spectrum:
        system.cavity_model.plot_analytical_polaritonic_frequencies(matter_freqs, fig, ax, positive_only=True)
    else:
        system.cavity_model.plot_numerical_polaritonic_frequencies(matter_freqs, fig, ax)
    format_input_output_plot_matter_freq(fig, ax, title=title)

    if show:
        plt.show()

    return fig, ax


def polaritonic_freqs_plot(matter_range, spectrum, fig=None, ax=None, label_prefix='', label_suffix='', **plot_kwargs):
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1)

    for i, polaritonic_freq in enumerate(spectrum):
        ax.plot(matter_range, polaritonic_freq, label=label_prefix + ("$f_{%d}$" % i) + label_suffix, **plot_kwargs)
    ax.set_xlabel("Matter frequency $f_b$")
    ax.set_xlim(matter_range[0], matter_range[-1])
    ax.set_ylabel("Polaritonic frequencies")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def transmission_slice_plot(f, s_param, legend_label=None, title=None, fig=None, ax=None, color=None, show=False):
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1)

    if color is None:
        ax.plot(f, s_param, label=legend_label)
    else:
        ax.plot(f, s_param, label=legend_label, color=color)

    if legend_label is not None:
        ax.legend()
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_xlim(f[0], f[-1])

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def input_output_plot(x, y, s_param, fig=None, ax=None, show=False, colorbar_label=None, dB=False, dB_step=10,
                      colorbar_lims=None, colorbar_ticks=None, colorbar_format='%0.1f', colormap=input_output_cmap):
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1)

    ax.grid(False)
    if colorbar_lims is None:
        if not dB:
            colorbar_lims = (0, 1)
        else:
            colorbar_lims = (np.min(s_param), np.max(s_param))
    c = ax.pcolormesh(x, y, s_param, shading='auto',
                      vmin=colorbar_lims[0], vmax=colorbar_lims[1], cmap=colormap)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xlabel(r"Matter mode(s) frequency $\omega_b/(2\pi)$ (GHz)")
    ax.set_ylabel(r"Frequency $\omega/(2\pi)$ (GHz)")

    if colorbar_label is None:
        if not dB:
            colorbar_label = "$|S_{21}|^2$"
        else:
            colorbar_label = "$|S_{21}|^2$ (dB)"

    ax.grid(False)
    if (colorbar_ticks is None) and (dB is not None) and dB:
        # Smart colorbar ticks: display min and max, but round uniformly between
        a, b = round(colorbar_lims[0], -1), round(colorbar_lims[1], -1)
        colorbar_ticks = np.arange(a, b+dB_step, dB_step)
        if colorbar_lims[0] not in colorbar_ticks:
            colorbar_ticks = np.append(np.array([colorbar_lims[0]]), colorbar_ticks)
        if colorbar_lims[1] not in colorbar_ticks:
            colorbar_ticks = np.append(colorbar_ticks, np.array([colorbar_lims[1]]))
    fig.colorbar(c, ax=ax, label=colorbar_label, pad=0.05, ticks=colorbar_ticks)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def format_input_output_plot_matter_freq(fig, ax, title=None):
    ax.set_xlabel("Matter frequency (GHz)")
    ax.set_ylabel("Frequency drive (GHz)")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


def format_input_output_plot_magnetic_field(fig, ax, title=None):
    ax.set_xlabel(r"Applied magnetic field $\mu_0 H$ (mT)")
    ax.set_ylabel("Frequency input (GHz)")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


def plot_prob_density_1d(x, psi, title=None):
    plt.plot(x, np.abs(psi)**2)
    plt.xlim(x[0], x[-1])
    if title is None:
        plt.title("Probability density")
    else:
        plt.title(title)
    plt.show()


def prob_density_anim_1d(x, psi, time_grid, blit=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel(r'$|\psi(x)|^2$')
    title = ax.set_title("")
    prob_line, = ax.plot([], [])

    def init():
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 0.04)
        title.set_text("")
        prob_line.set_data([], [])
        return prob_line, title

    def update(t):
        prob_line.set_data(x, np.abs(psi[:, t]) ** 2)
        title.set_text("t = %.2f" % time_grid.span[t])
        return prob_line, title

    animation = FuncAnimation(fig, update, init_func=init, frames=time_grid.length, interval=1, blit=blit)
    return animation


def plot_prob_density_2d(grid, wf, title=None):
    prob_density = abs(wf.reshape(grid.Nx, grid.Nz))**2
    prob_density = prob_density.T / prob_density.max()

    fig, ax = plt.subplots()
    c = ax.pcolor(grid.x, grid.z, prob_density, cmap='jet', vmin=0, vmax=1)

    if title is None:
        ax.set_title('Probability density')
    else:
        ax.set_title(title)

    fig.colorbar(c, ax=ax)


def prob_density_anim_2d(grid, wf, time_grid: Grid1D, dt=0.5, interval=100, blit=False):
    def prob_density(t):
        prob_density = abs(wf[:, t].reshape(grid.Nx, grid.Nz)) ** 2
        return prob_density[:-1, :-1].T / prob_density.max()

    def init():
        ax.set_xlim(grid.x_start, grid.x_end)
        ax.set_ylim(grid.z_start, grid.z_end)
        title.set_text("")
        quad.set_array([])
        return title, quad

    def update(t):
        # Find closest value in the time grid that correspond to the displayed time
        t = time_grid.closest_index(displayed_times[t])

        # Update title
        title.set_text("t = %.2f" % time_grid.span[t])

        # Update probability density
        p = prob_density(t)
        quad.set_array(p.ravel())
        return title, quad

    displayed_times = np.arange(time_grid.start, time_grid.end + dt, dt)

    fig, ax = plt.subplots()
    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel(r'$z$ (atoms)')
    # ax.set_ylabel(r'$|\psi(x)|^2$')
    title = ax.set_title("")

    quad = ax.pcolor(grid.x, grid.z, prob_density(0), cmap='jet', vmin=0, vmax=1)
    cb = fig.colorbar(quad, ax=ax)

    animation = FuncAnimation(fig, update, init_func=init, frames=len(displayed_times), interval=interval, blit=blit)
    return animation


def plot_occupation_prob(coefficients, time_grid, title=None):

    for n in range(len(coefficients)):
        plt.plot(time_grid.span, np.abs(coefficients[n, :])**2, label="E_{%d}" % n)

    plt.xlim(time_grid.start, time_grid.end)
    plt.xlabel('$t$ (ps)')
    plt.ylabel(r'$|c_n(t)|^2$')
    if title is None:
        plt.title("Time evolution of occupation probabilities")
    else:
        plt.title(title)
    plt.show()

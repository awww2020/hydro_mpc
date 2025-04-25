import logging
import os

# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation

FIGURES_PATH = os.path.join('./figures')
FIGURE_WIDTH_PT = 396

# 设置中文字体为宋体
# plt.rcParams['font.sans-serif'] = ['SimSun']   # 使用宋体
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块的问题

def fig_size(width_pt=FIGURE_WIDTH_PT, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Returns the width and heights in inches for a matplotlib figure.

    :param float width_pt: document width in points, in latex can be determined with \showthe\linewidth
    :param float fraction: fraction of the width with which the figure will occupy
    :param float ratio: ratio of the figure, default is the golden ratio
    :param tuple subplots: the shape of subplots
    :return: float fig_width_in: width in inches of the figure, float fig_height_in: height in inches of the figure
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def new_fig(width_pt=396, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Creates new instance of a `matplotlib.pyplot.figure` fig by using the `fig_size` function.

    :param float width_pt: document width in points, in latex can be determined with \showthe\textwidth
    :param float fraction: fraction of the width with which the figure will occupy
    :param float ratio: ratio of the figure, default is the golden ratio
    :param tuple subplots: the shape of subplots
    :return: matplotlib.pyplot.figure fig: instance of a `matplotlib.pyplot.figure` with desired width and height
    """
    fig = plt.figure(figsize=fig_size(width_pt, fraction, ratio, subplots))
    return fig


def save_fig(fig, name, path=None, tight_layout=True):
    """
    Saves a `matplotlib.pyplot.figure` as pdf file.

    :param matplotlib.pyplot.figure fig: instance of a `matplotlib.pyplot.figure` to save
    :param str name: filename without extension
    :param str path: path where the figure is saved, if None the figure is saved at the results directory
    :param bool crop: bool if the figure is cropped before saving
    """
    if tight_layout:
        fig.tight_layout()

    if path is None:
        path = FIGURES_PATH

    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(os.path.join(path, f'{name}.eps'))
    fig.savefig(os.path.join(path, f'{name}.png'))

def animate(Y_test, Y_preds, labels, fraction=1, fps=100, save_ani=False):
    pass



def plot_input_sequence(T, U, filename=None):
    M = U.shape[1]
    linewidth = 1
    fig = new_fig()
    ax = fig.add_subplot(111)
    ax.set(xlim=[np.min(T), np.max(T)])
    ax.set(xlabel='Time $t$ (s)', ylabel=fr'Input $u$ (m3/s)')
    colors = ['tab:blue',  'tab:gray','tab:red']
    for i in range(M):
        if i == 0 or i == 2:
            # ax.step(T, U[:, i], where='post', linewidth=linewidth, label=fr'$u_{i + 1}$', c=colors[i])
            ax.plot(T, U[:, i], linewidth=linewidth, label=fr'$u_{i + 1}$', c=colors[i])
    ax.legend(loc='best')
    ax.grid('on')
    fig.tight_layout()
    if filename is not None:
        save_fig(fig, filename)
    plt.show(block=True)


def plot_states_z(T, Z_ref, Z_pred=None, Z_mpc=None, filename=None):
    linewidth = 1
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ (s)', ylabel=r'Water level $z$ (m)')
    ax.set(xlim=[np.min(T), np.max(T)])
    colors_ref = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:gray']
    colors_pred_mpc = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:gray']

    for i, x in enumerate(['0km', '1km', '2km', '3km', '4km']):
        ref_label = rf'${x}^{{\mathrm{{ref}}}}$'
        ax.plot(T, Z_ref[:, i], linewidth=linewidth, label=ref_label, c=colors_ref[i])
        if Z_pred is not None:
            pred_label = rf'$\widehat{{{x}}}$'
            ax.plot(T, Z_pred[:, i], linestyle='--', linewidth=linewidth, label=pred_label, c=colors_pred_mpc[i])
        if Z_mpc is not None:
            mpc_label = rf'${x}^{{\mathrm{{MPC}}}}$'
            ax.plot(T, Z_mpc[:, i], linestyle='--', linewidth=linewidth, label=mpc_label, c=colors_pred_mpc[i])


    ax.grid('on')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    if filename is not None:
        save_fig(fig, filename)
    fig.tight_layout()
    plt.show(block=True)

def plot_states_u(T, Z_ref, Z_pred=None, Z_mpc=None, filename=None):
    '''
    画流速的变化
    '''
    linewidth = 1
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ (s)', ylabel=r'Velocity $u$ (m/s)')
    ax.set(xlim=[np.min(T), np.max(T)])
    colors_ref = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:gray']
    colors_pred_mpc = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:gray']

    for i, x in enumerate(['0km', '1km', '2km', '3km', '4km']):
        ref_label = rf'${x}^{{\mathrm{{ref}}}}$'
        ax.plot(T, Z_ref[:, i], linewidth=linewidth, label=ref_label, c=colors_ref[i])
        if Z_pred is not None:
            pred_label = rf'$\widehat{{{x}}}$'
            ax.plot(T, Z_pred[:, i], linestyle='--', linewidth=linewidth, label=pred_label, c=colors_pred_mpc[i])
        if Z_mpc is not None:
            mpc_label = rf'${x}^{{\mathrm{{MPC}}}}$'
            ax.plot(T, Z_mpc[:, i], linestyle='--', linewidth=linewidth, label=mpc_label, c=colors_pred_mpc[i])

    ax.grid('on')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    if filename is not None:
        save_fig(fig, filename)
    fig.tight_layout()
    plt.show(block=True)

def plot_aim_states(T, Z_ref, Z_pred=None, Z_mpc=None, filename=None):
    linewidth = 1
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ (s)', ylabel=r'Water level $z$ (m)')
    ax.set(xlim=[np.min(T), np.max(T)])
    colors_ref = 'tab:blue'
    colors_pred_mpc = 'tab:orange'

    ref_label = rf'$4km^{{\mathrm{{ref}}}}$'
    ax.plot(T, Z_ref, linewidth=linewidth, label=ref_label, c=colors_ref)
    if Z_pred is not None:
        pred_label = rf'$\widehat{{4km}}$'
        ax.plot(T, Z_pred, linestyle='--', linewidth=linewidth, label=pred_label, c=colors_pred_mpc)
    if Z_mpc is not None:
        mpc_label = rf'$4km^{{\mathrm{{MPC}}}}$'
        ax.plot(T, Z_mpc[:, -1], linestyle='--', linewidth=linewidth, label=mpc_label, c=colors_pred_mpc)

    ax.grid('on')
    ax.legend(loc='best')
    if filename is not None:
        save_fig(fig, filename)
    fig.tight_layout()
    plt.show(block=True)


def plot_absolute_error_z(T, Z_ref, Z_pred=None, Z_mpc=None, filename=None):
    '''
    画水位绝对误差
    :param T:
    :param Z_ref:
    :param Z_pred:
    :param Z_mpc:
    :param filename:
    :return:
    '''
    linewidth = 1
    states = ['0km', '1km', '2km', '3km', '4km']
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:gray']
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ (s)', ylabel=r'absolute_error (m)')
    ax.grid(True, which='major')
    ax.set(xlim=[np.min(T), np.max(T)])

    for i, state in enumerate(states):
        abs_errors_list = []
        if Z_pred is not None:
            abs_errors = np.abs(Z_ref[:, i] - Z_pred[:, i])
            label = fr'${states[i]}$'
        else:
            abs_errors = np.abs(Z_ref[:, i] - Z_mpc[:, i])
            label = fr'${states[i]}$'

        abs_errors_list.append(abs_errors)

        mae = abs_errors.mean()
        maxe = abs_errors.max()
        ax.plot(T, abs_errors, linewidth=linewidth,
                label=label, c=colors[i])
        logging.info(label + f', MAE: {mae:.2e}, MaxE: {maxe:.2e}')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    if filename is not None:
        save_fig(fig, filename)
    fig.tight_layout()
    plt.show(block=True)

def plot_absolute_error_u(T, Z_ref, Z_pred=None, Z_mpc=None, filename=None):
    '''
    画流速绝对误差
    :param T:
    :param Z_ref:
    :param Z_pred:
    :param Z_mpc:
    :param filename:
    :return:
    '''
    linewidth = 1
    states = ['0km', '1km', '2km', '3km', '4km']
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:gray']
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ (s)', ylabel=r'absolute_error (m/s)')
    ax.grid(True, which='major')
    ax.set(xlim=[np.min(T), np.max(T)])

    for i, state in enumerate(states):
        abs_errors_list = []
        if Z_pred is not None:
            abs_errors = np.abs(Z_ref[:, i] - Z_pred[:, i])
            label = fr'${states[i]}$'
        else:
            abs_errors = np.abs(Z_ref[:, i] - Z_mpc[:, i])
            label = fr'${states[i]}$'

        abs_errors_list.append(abs_errors)

        mae = abs_errors.mean()
        maxe = abs_errors.max()
        ax.plot(T, abs_errors, linewidth=linewidth,
                label=label, c=colors[i])
        logging.info(label + f', MAE: {mae:.2e}, MaxE: {maxe:.2e}')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    if filename is not None:
        save_fig(fig, filename)
    fig.tight_layout()
    plt.show(block=True)

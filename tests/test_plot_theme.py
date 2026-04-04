import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from resources.graphics.plot_theme import apply_plot_theme


def test_apply_plot_theme_sets_color_cycle():
    apply_plot_theme()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [mcolors.to_hex(c['color']) for c in prop_cycle]
    assert colors[0] == '#4caf50'
    assert colors[1] == '#2196f3'


def test_apply_plot_theme_sets_figure_size():
    apply_plot_theme()
    assert plt.rcParams['figure.figsize'] == [10, 6]


def test_apply_plot_theme_sets_white_background():
    apply_plot_theme()
    assert plt.rcParams['figure.facecolor'] == 'white'
    assert plt.rcParams['axes.facecolor'] == 'white'


def test_apply_plot_theme_spine_visibility():
    apply_plot_theme()
    assert plt.rcParams['axes.spines.top'] == False
    assert plt.rcParams['axes.spines.right'] == False


def test_apply_plot_theme_grid():
    apply_plot_theme()
    assert plt.rcParams['axes.grid'] == True
    assert plt.rcParams['grid.color'] == '#f5f5f5'

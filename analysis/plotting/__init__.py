import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from snr.constants import ROOT_DIR

AI2_TEAL = "#212121"
AI2_OFF_WHITE = "#FAF2E9"


def init_manrope():
    manrope_font_path = ROOT_DIR / "analysis" / "plotting" / "manrope.ttf"
    fm.fontManager.addfont(str(manrope_font_path.absolute()))
    plt.rcParams["font.family"] = "Manrope"
    plt.rcParams["text.color"] = AI2_TEAL


def add_white_background(ax: plt.Axes):
    """Add a background rectangle"""
    ax.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes, facecolor=AI2_OFF_WHITE, edgecolor="none", zorder=-1
        )
    )
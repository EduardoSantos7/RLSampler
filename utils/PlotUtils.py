from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


class PlotUtils:

    @staticmethod
    def plot_lines(xs, ys, labels):
        if len(labels) != len(ys):
            raise Exception("Different sizes in params")

        for x, y, label in zip(xs, ys, labels):
            plt.plot(x, y, label=label)

        plt.show()

    @staticmethod
    def different_axis(episodes, rewards, sample_size, epsilon):
        host = host_subplot(111, axes_class=AA.Axes)

        plt.subplots_adjust(right=0.75)

        par1 = host.twinx()

        par1.axis["right"].toggle(all=True)

        host.set_xlim(0, len(episodes))
        host.set_ylim(0, 1)

        host.set_xlabel("Episodes")
        host.set_ylabel("Rewards")
        par1.set_ylabel("Sample Size")

        p1, = host.plot(episodes, rewards, label="Rewards")
        p2, = par1.plot(episodes, sample_size, label="Sample Size")

        par1.set_ylim(1, 350)

        host.legend()

        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())

        plt.draw()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MeasurementData:
    def __init__(self, fname):
        self.df = pd.read_csv(fname)

    def plot_attr(self, t: float, *attr, save = True):
        t_close = self.df['t'][abs(self.df['t'] - t).argmin()]
        data_i = self.df[self.df['t'] == t_close]

        fname = "plot_"
        
        for var in attr:
            plt.scatter(data_i['x'].to_numpy(),
                        data_i[var].to_numpy(),
                        label = var,
                        s = 2)
            fname += var + "_"

        plt.legend()
        plt.title(f"t = {t_close}")
        plt.tight_layout()
        if(save):
            plt.savefig(fname + str(t_close) + ".pdf", dpi = 300)
        else:
            plt.show()

        plt.clf()

if __name__ == "__main__":
    md = MeasurementData("data/visc.dat")
    md.plot_attr(0.1, 'pressure', 'density')
    md.plot_attr(0.1, 'velocity')

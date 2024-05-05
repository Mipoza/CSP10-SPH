import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import re
from collections import defaultdict
import pandas as pd
plt.rcParams.update({
    "text.usetex": True })

class RuntimeData(pd.DataFrame):
    def __init__(self, filename, *args, **kwargs):
        df = self.parse_from_out(filename, *args, **kwargs)
        super(RuntimeData, self).__init__(df)
        self.df = df
        
    def rtimes_pd(self, threads, resolution):
        return self.df[(self.df["threads"] == threads) &
                       (self.df["resolution"] == resolution)]\
                .drop(columns = ["threads", "resolution", "cg_it"])
    def rtimes(self, threads, resolution):
         return np.array(
             self.rtimes_pd(threads, resolution)\
                 .values\
                 .T)
             
    def cg_it(self, threads, resolution):
        return np.mean(self.df[(self.df["threads"] == threads) &
                       (self.df["resolution"] == resolution)]\
                .drop(columns = ["threads", "resolution", "runtime"])\
                .values\
                .T) # they should all be the same (+- a few)

    def parse_from_out(self, filename, no_experiments = 8, no_res_idx = 5,
                       base_res_exp = 6, no_thread_exp = 6, warmup = True):
        data_pattern = re.compile(r"(?<=(###)) *.* (?=(###))")
        file = open(filename)
        content = file.read()
        matches = data_pattern.finditer(content)
        
        runtimes = pd.DataFrame({"threads": [],
                                 "resolution": [],
                                 "runtime": [],
                                 "cg_it": []})
        warmed_up = {}
        warmed_up = defaultdict(lambda: not warmup, warmed_up)
        
        for match in matches:
            # Convert to array of floats
            array_result = list(map(float, match.group().split(', ')))
            
            threads = int(array_result[0])
            res = int(array_result[1])
            runtime = array_result[-1]
            cg_it = int(array_result[3])
            
            if(warmed_up[threads, res]):
                new = pd.Series({"threads": threads,
                                 "resolution": res,
                                 "runtime": runtime,
                                 "cg_it": cg_it})
                runtimes = pd.concat([runtimes, new.to_frame().T],
                                ignore_index=True)
            else:
                warmed_up[threads, res] = True
            
        return runtimes

if __name__ == "__main__":
    data = RuntimeData("fisher-54829997.out")
    nstd = 1.
    
    # Runtimes for the strong scaling task
    # Standard deviations include in plot
    res = np.array([64, 128, 256, 512, 1024])
    th = [1, 2, 4, 8, 16]
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    ax1.set_xlabel("No. Threads")
    ax1.set_ylabel("Time [s]")
    ax2.set_xlabel("No. Threads")
    ax2.set_ylabel("Speedup")
    for r in res:
        rtimes = np.array([np.mean(data.rtimes(t, r)) for t in th])
        stds = np.array([np.std(data.rtimes(t, r)) for t in th])
        ax1.loglog(th, rtimes,
                label = str(r), marker = (8, 2), markersize = 6)
        ax1.fill_between(th, rtimes - nstd*stds,
                         rtimes + nstd*stds, alpha = 0.2)
        
        speedups = np.array([
            np.mean(np.outer(data.rtimes(1, r), 1./data.rtimes(t, r)))
            for t in th])
        stds = np.array([
            np.std(np.outer(data.rtimes(1, r), 1./data.rtimes(t, r)))
            for t in th])
        ax2.plot(th, speedups, label = str(r), marker = (8, 2), markersize = 6)
        ax2.fill_between(th, speedups - nstd*stds,
                         speedups + nstd*stds, alpha = 0.2)
    ax1.set_xticks(th)
    ax1.tick_params(which='minor', length=0)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.plot(th, th, label = "Optimal", color = 'black')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.legend()
    fig2.legend(loc = 'upper center')
    fig1.savefig("fisher_runtimes_strong.pdf")
    fig2.savefig("fisher_scaling_strong.pdf")
    plt.show()
    plt.clf()
    
    # Plots for weak scaling task
    res = np.array([64, 128, 256, 512, 1024, 2048])
    th = [1, 4, 16, 64]
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    ax1.set_xlabel("n")
    ax1.set_ylabel("Time [s]")
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Efficiency")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    for offset in range(3):
        rtimes = np.array([np.mean(data.rtimes(t, r)) for (t, r)
                           in zip(th, res[offset:offset+4])])
        stds = np.array([np.std(data.rtimes(t, r)) for (t, r)
                         in zip(th, res[offset:offset+4])])
        ax1.loglog(res[offset:offset+4], rtimes, label = str(res[offset]), 
                   marker = (8, 2), markersize = 6)
        ax1.fill_between(res[offset:offset+4],
                         rtimes - nstd*stds, rtimes + nstd*stds, alpha = 0.2)
        
        efficiency = np.array([np.mean(np.outer(data.rtimes(1, res[offset]), 
                                                1./data.rtimes(t, r)))
                               for (t, r) in zip(th, res[offset:offset+4])])
        stds = np.array([np.std(np.outer(data.rtimes(1, res[offset]), 
                                                1./data.rtimes(t, r))) 
                         for (t, r) in zip(th, res[offset:offset+4])])
        ax2.semilogx(th, efficiency, label = str(res[offset]),
                      marker = (8, 2), markersize = 6)
        ax2.fill_between(th, efficiency - nstd*stds, 
                             efficiency + nstd*stds, alpha = 0.2)
    ax1.set_xticks(res)
    ax1.tick_params(which='minor', length=0)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.set_xticks(th)
    ax2.tick_params(which='minor', length=0)
    ax2.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.plot(th, np.ones_like(th), label = "Optimal", color = 'black')
    fig1.legend(loc = 'upper left')
    fig1.tight_layout()
    fig2.legend()
    fig2.tight_layout()
    fig1.savefig("fisher_runtimes_weak.pdf")
    fig2.savefig("fisher_scaling_weak.pdf")
    plt.show()
    plt.clf()
    
    # Weak scaling normalized to cg iterations
    res = np.array([64, 128, 256, 512, 1024, 2048])
    th = [1, 4, 16, 64]
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    ax1.set_xlabel("n")
    ax1.set_ylabel("Time per CG-Iteration [s]")
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Efficiency")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    
    for offset in range(3):
        # runtimer PER ITERATION
        rtimes = np.array([np.mean(data.rtimes(t, r)) for (t, r)
                           in zip(th, res[offset:offset+4])])
        cg_its = np.array([data.cg_it(t, r) for (t, r)
                           in zip(th, res[offset:offset+4])])
        rtimes = rtimes/cg_its
        stds = np.array([np.std(data.rtimes(t, r)) for (t, r)
                         in zip(th, res[offset:offset+4])])
        ax1.errorbar(res[offset:offset+4], rtimes, yerr = stds,
                     label = str(res[offset]), marker = (8, 2), markersize = 6,
                     capsize = 8, linewidth = 0.5)
        # ax1.loglog(res[offset:offset+4], rtimes, label = str(res[offset]), 
        #             marker = (8, 2), markersize = 6)
        ax1.fill_between(res[offset:offset+4], 
                          rtimes - nstd*stds, 
                          rtimes + nstd*stds, alpha = 0.075)
        # ax1.errorbar(res[offset:offset+4], rtimes, yerr = nstd*stds, 
                     # label = str(res[offset]), marker = (8, 2), markersize = 6)
        
        efficiency = np.array([np.mean(np.outer(data.rtimes(1, res[offset])\
                                                /data.cg_it(1, res[offset]), 
                                                data.cg_it(t, r)\
                                                /data.rtimes(t, r)))
                               for (t, r) in zip(th, res[offset:offset+4])])
        stds = np.array([np.std(np.outer(data.rtimes(1, res[offset])\
                                                /data.cg_it(1, res[offset]), 
                                                data.cg_it(t, r)\
                                                /data.rtimes(t, r))) 
                         for (t, r) in zip(th, res[offset:offset+4])])
        ax2.errorbar(th, efficiency, yerr = stds, label = str(res[offset]),
                      marker = (8, 2), markersize = 6,
                      capsize = 8, linewidth = 0.8)
        # ax2.semilogx(th, efficiency, label = str(res[offset]),
        #               marker = (8, 2), markersize = 6)
        ax2.fill_between(th, efficiency - nstd*stds, 
                              efficiency + nstd*stds, alpha = 0.1)
    ax1.set_xticks(res)
    ax1.set_xlim((60, 2500))
    ax1.tick_params(which='minor', length=0)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.set_xticks(th)
    ax2.tick_params(which='minor', length=0)
    ax2.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.plot(th, np.ones_like(th), label = "Optimal", color = 'black')
    fig1.legend(loc = 'upper left')
    fig1.tight_layout()
    fig2.legend()
    fig2.tight_layout()
    fig1.savefig("fisher_runtimes_per_it_weak.pdf")
    fig2.savefig("fisher_scaling_per_it_weak.pdf")
    plt.show()
    plt.clf()
    
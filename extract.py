import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=22)
plt.rcParams.update({'legend.fontsize': 19, 'xtick.labelsize': 19,
'ytick.labelsize': 19, 'axes.labelsize': 19})

def load_data(basename):
    files_no_transfer = [open("results/{0}_frac_{1}/diagnostics.txt".format(basename, frac)).read() for frac in ["0.10", "0.50","1.0"] ]
    
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", file))) for file in files_no_transfer]
    valid = [acc[1::2] for acc in accuracies]
    return np.array(valid).astype(np.float32)

f = plt.figure(figsize=(15, 15))
basename_to_plotname = {"domainB": "No transfer learning. Fraction = ", "transfer": "With transfer learning. Fraction = "}
plot_type = {"domainB": "--", "transfer": "-"}
colors  = sns.color_palette(n_colors = 4)

for basename in ["domainB", "transfer"]:
    current_palette = sns.color_palette()
    vt = load_data(basename)
    x_ticks = range(vt.shape[1])
    legends  = map( lambda x: basename_to_plotname[basename] + x ,["0.10", "0.50", "1.0"])
    for values, legend_name, color in zip(vt, legends, colors):
        plt.plot(x_ticks, values, plot_type[basename], label=legend_name, color=color)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

plt.xticks(x_ticks[49::50], map(lambda x: x + 1, x_ticks[49::50]))
#f.suptitle("Evalutating transfer properties")
plt.legend(loc = 8)
plt.savefig("figs/transfer_results.pdf")

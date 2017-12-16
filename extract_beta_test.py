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
plt.rcParams.update({'legend.fontsize': 16, 'xtick.labelsize': 16,
'ytick.labelsize': 16, 'axes.labelsize': 16})
def load_data(basename):
    files_no_transfer = [open("results/{0}_beta_{1}/diagnostics.txt".format(basename, beta_type)).read() for beta_type in ["Blundell", "Standard", "None"] ]
    
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", file))) for file in files_no_transfer]
    valid = [acc[1::2] for acc in accuracies]
    return np.array(valid).astype(np.float32)

f = plt.figure(figsize=(10, 8))

current_palette = sns.color_palette()
vt = load_data("[0,1,2,3,4,5,6,7,8,9]")
x_ticks = range(vt.shape[1])
colors  = sns.color_palette(n_colors = 4)
legends  = [r"$\frac{2^{M-i}}{2^M-1}$", r"$\frac{1}{M}$", "0"]
for values, legend_name, color in zip(vt,legends, colors):
    plt.plot(x_ticks, values, label=r"Accuracy with transfer, with $\beta$= {0}".format(legend_name), color=color)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(x_ticks[::5], map(lambda x: x+1, x_ticks[::5]))
f.suptitle("Accuracy after training for 50 epochs")
plt.legend(loc = 7)

plt.savefig("figs/comparing_beta.pdf")



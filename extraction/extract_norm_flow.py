import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=24)
plt.rcParams.update({'legend.fontsize': 24, 'xtick.labelsize': 24,
'ytick.labelsize': 24, 'axes.labelsize': 24})
def load_data(basename):
    normflow = open("results/{}/diagnostics.txt".format(basename)).read()
    
    accuracies = list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", normflow)))
    valid = accuracies[1::2]
    return np.array(valid).astype(np.float32)

f = plt.figure(figsize=(10, 8))

current_palette = sns.color_palette()

colors = sns.color_palette(n_colors = 2)
basenames = ["beta_blundell", "normflow"]
legends = ["Without Normalizing Flows", "With Normalizing Flows"]
for basename, color, legend in zip(basenames, colors, legends):
    vt = load_data(basename)
    x_ticks = range(vt.shape[0])
    plt.plot(x_ticks, vt, label=legend, color=color)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(x_ticks[59::60], map(lambda x: x + 1 , x_ticks[59::60]))
#f.suptitle("With and without Normalizing Flows")
plt.legend(loc = 8)

plt.savefig("figs/normflow.pdf")



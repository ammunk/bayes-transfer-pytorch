import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np

plt.rc('font', family='serif', size=22)
plt.rcParams.update({'legend.fontsize': 16, 'xtick.labelsize': 16,
'ytick.labelsize': 16, 'axes.labelsize': 16})
def load_data(basename):
    normflow = open("norm_flow/{}.txt".format(basename)).read()
    
    accuracies = list(map(lambda x: x.split(" ")[-1], re.findall(r"(acc: \d.\d+)", normflow)))
    valid = accuracies[1::3]
    if basename=="beta_blundell":
        valid=valid[:50]
    return np.array(valid).astype(np.float32)

f = plt.figure(figsize=(10, 8))

current_palette = sns.color_palette()

colors = sns.color_palette(n_colors = 2)
basenames = ["beta_blundell", "logfile"]
legends = ["Without Normalizing Flows", "With Normalizing Flows"]
for basename, color, legend in zip(basenames, colors, legends):
    vt = load_data(basename)
    x_ticks = range(vt.shape[0])
    plt.plot(x_ticks, vt, label=legend, color=color)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(x_ticks[5::5], map(lambda x: x, x_ticks[5::5]))
f.suptitle("With and without Normalizing Flows")
plt.legend(loc = 7)

plt.savefig("figs/normflow.pdf")



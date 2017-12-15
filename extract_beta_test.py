import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np

def load_data(basename):
    files_no_transfer = [open("results/{0}_beta_{1}/logfile.txt".format(basename, beta_type)).read() for beta_type in ["blundell", "ole", "naive", "0"] ]
    
    
    
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(acc: \d.\d+)", file))) for file in files_no_transfer]
    valid = [acc[1::3] for acc in accuracies]
    
    return np.array(valid).astype(np.float32)

f = plt.figure(figsize=(10, 8))

current_palette = sns.color_palette()
vt = load_data("transfer_domain1all")
x_ticks = range(vt.shape[1])
colors  = sns.color_palette(n_colors = 4)
legends  = ["blundell", "ole", "naive", "0"]
for values, legend_name, color in zip(vt,legends, colors):
    plt.plot(x_ticks, values, label=r"Accuracy with transfer, with $\beta$ = {0}".format(legend_name), color=color)

vnt = load_data("domain1all")
for values, legend_name, color in zip(vnt,legends, colors): 
    plt.plot(x_ticks, values, "--", label=r"Accuracy without transfer, with $\beta$ = {0}".format(legend_name), color=color)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(x_ticks[::5], map(lambda x: x+1, x_ticks[::5]))
f.suptitle("Accuracy after training for 50 epochs")
plt.legend()

plt.savefig("figs/comparing_beta.pdf")



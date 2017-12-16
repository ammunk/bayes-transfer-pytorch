import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np

def load_data(basename, intervals, name_ext):
    files = [open("results/{}{}{}/logfile.txt".format(basename, i, name_ext)).read() for i in intervals]
    acc = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(acc: \d.\d+)", f))) for f in files]
    train = list(map(lambda x: x[-3], acc))
    valid = list(map(lambda x: x[-2], acc))
    MAP = list(map(lambda x: x[-1], acc))
    return np.array(train).astype(np.float32), np.array(valid).astype(np.float32), np.array(MAP).astype(np.float32)

i = [0.05, 0.1, 0.2, 0.3, 0.5, 1]
f = plt.figure(figsize=(10, 8))

	
plot_types  = ["all", "3869", "1725"]
colors  = sns.color_palette(n_colors = 3)
legends = [r"$(1,7)\rightarrow(2,5)$", r"$(3,8)\rightarrow(6,9)$",r"$(0-4)\rightarrow(5-9)$"]
for name_ext, color, legend in zip(plot_types, colors, legends):
    train, valid, MAP = load_data("transfer_domain", i, name_ext)
    plt.plot(i, train, "--", label= legend +r"Train, prior: $q(w \mid \theta)$", color=color)

for name_ext, color, legend in zip(plot_types, colors, legends):
    train, valid, MAP = load_data("domain", i, name_ext)
    plt.plot(i, valid, label=legend+r"Validation, prior: $\mathcal{U}(a, b)$", color=color)

plt.xlabel("Size of transfer dataset")
plt.ylabel("Accuracy")
plt.xticks(i, map(lambda x: "{}%".format(int(x*100)), i))
f.suptitle("Accuracy after training for 50 epochs")
plt.legend()


plt.savefig("figs/transfer_result.pdf")


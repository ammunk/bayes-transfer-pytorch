import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import re
import numpy as np

def load_data(basename, intervals, n_digits):
    files = [open("results/{}{}{}/logfile.txt".format(basename, i, n_digits)).read() for i in intervals]
    acc = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(acc: \d.\d+)", f))) for f in files]
    if basename is "domain":
        print(acc)
    train = list(map(lambda x: x[-3], acc))
    valid = list(map(lambda x: x[-2], acc))
    MAP = list(map(lambda x: x[-1], acc))
    return np.array(train).astype(np.float32), np.array(valid).astype(np.float32), np.array(MAP).astype(np.float32)

i = [0.05, 0.1, 0.2, 0.3, 0.5, 1]
f = plt.figure(figsize=(10, 8))
all_used = False
if all_used:
	n_digits = "all"
else:
	n_digits = "two"

train, valid, MAP = load_data("transfer_domain", i, n_digits)

#plt.plot(i, train, label=r"Train, prior: $q(w \mid \theta)$", color="#9c209b")
plt.plot(i, valid, "--", label=r"Validation, prior: $q(w \mid \theta)$", color="#d534d3")
plt.plot(i, MAP, "--", label=r"MAP, prior: $q(w \mid \theta)$", color="#e273e1")

train, valid, MAP = load_data("domain", i, n_digits)

#plt.plot(i, train, label=r"Train, prior: $\mathcal{U}(a, b)$", color="#209c22")
plt.plot(i, valid, "--", label=r"Validation, prior: $\mathcal{U}(a, b)$", color="#34d536")
plt.plot(i, MAP, "--", label=r"MAP, prior: $\mathcal{U}(a, b)$", color="#73e275")

plt.xlabel("Size of transfer dataset")
plt.ylabel("Accuracy")
plt.xticks(i, map(lambda x: "{}%".format(int(x*100)), i))
f.suptitle("Accuracy after training for 50 epochs")
plt.legend()

plt.savefig("figs/" + n_digits + "result.pdf")


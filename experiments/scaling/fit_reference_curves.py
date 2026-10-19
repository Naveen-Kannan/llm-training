import json
from collections import defaultdict
import scipy
import matplotlib.pyplot as plt
import numpy as np

with open("results/scaling/reference_isoflops_curves.json", "r") as f:
    data = json.load(f)

L_C = defaultdict(lambda: (0, 0, float("inf")))

for run in data:
    N = run["parameters"]
    C = run["compute_budget"]
    D = C / (6 * N)
    L = run["final_loss"]

    if L < L_C[C][2]:
        L_C[C] = (N, D, L)


def scaling_law(C, A, P):
    return A * C**P


########## Compute Optimal Model Size ##########
CN = [(C, N) for C, (N, D, L) in L_C.items()]

N_C_curve = scipy.optimize.curve_fit(scaling_law, [cn[0] for cn in CN], [cn[1] for cn in CN])

Cs = np.logspace(18, 24, 100)
plt.figure()
plt.scatter([cn[0] for cn in CN], [cn[1] for cn in CN])
plt.plot(Cs, scaling_law(Cs, *N_C_curve[0]), "r-")
plt.xscale("log")
plt.yscale("log")
plt.savefig("results/scaling/figures/model_size_scaling.png")

# Predicted Optimal Model Size
# 10 ** 23
print("C:10^23 -> N:", scaling_law(10**23, *N_C_curve[0]))
print("C:10^24 -> N:", scaling_law(10**24, *N_C_curve[0]))

########### Data Optimal Model Size ############
CD = [(C, D) for C, (N, D, L) in L_C.items()]

D_C_curve = scipy.optimize.curve_fit(scaling_law, [cd[0] for cd in CD], [cd[1] for cd in CD])

plt.figure()
plt.scatter([cd[0] for cd in CD], [cd[1] for cd in CD])
plt.plot(Cs, scaling_law(Cs, *D_C_curve[0]), "r-")
plt.xscale("log")
plt.yscale("log")
plt.savefig("results/scaling/figures/dataset_size_scaling.png")

# Predicted Optimal Data Size
# 10 ** 24
print("C:10^23 -> D:", scaling_law(10**23, *D_C_curve[0]))
print("C:10^24 -> D:", scaling_law(10**24, *D_C_curve[0]))

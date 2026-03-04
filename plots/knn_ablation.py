import matplotlib.pyplot as plt
import numpy as np
import scienceplots

n_best_actions = [1, 2, 4, 8, 16, 32, 64]
coverages = [7.07, 5.05, 14.44, 14.04, 14.64, 16.06, 15.35] 

plt.style.use(['science', 'nature', 'no-latex'])
fig, ax = plt.subplots(figsize=(8, 5))

x_pos = np.arange(len(n_best_actions))
bars = ax.bar(x_pos, coverages, color='skyblue', edgecolor='black', linewidth=0.8)

# --- NEW: HIGHLIGHT THE BASELINE AT 16 ---
baseline_index = n_best_actions.index(16)
baseline_bar = bars[baseline_index]

# 1. Change color and add hashing (dashed lines inside the bar)
baseline_bar.set_facecolor('steelblue') # Darker shade to distinguish it
baseline_bar.set_hatch('//')           # Diagonal lines pattern

# 2. Optional: Add a vertical dashed line for a clear boundary
# ------------------------------------------

for bar in bars:
    height = bar.get_height()
    ax.annotate(fr'${height}\%$', 
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12)

ax.set_ylim(0, 20)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.set_xlabel(r'$k$NN', fontsize=16)
ax.set_ylabel(r'Coverage $(\%)$', fontsize=16)

ax.set_xticks(x_pos)
ax.set_xticklabels([str(n) for n in n_best_actions])

ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('k_nn_ablation_baseline.pdf', dpi=300)
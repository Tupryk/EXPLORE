import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # Required to register the 'science' style

# Data provided
n_best_actions = [1, 2, 4, 8, 16, 32, 64]
coverages = [0.2, 6.06, 12.22, 14.14, 14.64, 16.17, 18.79] 

# Use SciencePlots styles
# 'science' sets the professional defaults
# 'ieee' or 'nature' are optional for specific journal formats
# 'no-latex' allows it to run without a local LaTeX installation
plt.style.use(['science', 'nature', 'no-latex'])

# Create the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Use categorical positions for bars
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

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    # Using 'fr' for raw f-string to avoid SyntaxWarning with \%
    ax.annotate(fr'${height}\%$', 
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12)

ax.set_ylim(0, 20)
ax.tick_params(axis='both', which='major', labelsize=14)
# Customizing axes and titles using LaTeX-style formatting
ax.set_xlabel(r'n-best actions', fontsize=14)
ax.set_ylabel(r'Coverage $(\%)$', fontsize=14)
#ax.set_title(r'Ablation Study: Impact of $n_{best\_actions}$ on Coverage', fontsize=14, pad=15)

# Set the x-ticks to correspond to n_best_action values
ax.set_xticks(x_pos)
ax.set_xticklabels([str(n) for n in n_best_actions])

# Add grid for readability (SciencePlots often turns this off, so we re-enable if desired)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Remove the top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('n_best_action_ablation_science.pdf', dpi=300)
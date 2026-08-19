"""
Stack PNG images vertically: Mechanism → WT → rad51
Creates a publication-ready composite figure
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import ammper_paths as P  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os

# ===== UPDATE THESE PATHS =====
# Path to your three PNG images
ASSETS = _os.path.join(P.RESULTS, "figures_updated_figures_branch")
MECHANISM_IMAGE = _os.path.join(ASSETS, "aBcartoon.PNG")
WT_GRAPHS = P.figures("WT_panel_all_doses.png")
RAD51_GRAPHS = P.figures("rad51_panel_all_doses.png")
# ==============================

# Check if files exist
for filepath in [MECHANISM_IMAGE, WT_GRAPHS, RAD51_GRAPHS]:
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please update the file paths at the top of the script")
        exit(1)

# Load images
img_mechanism = mpimg.imread(MECHANISM_IMAGE)
img_wt = mpimg.imread(WT_GRAPHS)
img_rad51 = mpimg.imread(RAD51_GRAPHS)

# Create figure with custom spacing
fig = plt.figure(figsize=(16, 22))

# Create gridspec with 3 rows, custom height ratios
gs = GridSpec(3, 1, figure=fig, 
              height_ratios=[1, 1.5, 1.5],  # Mechanism smaller, data plots larger
              hspace=0.05)  # Small gap between images

# Create subplots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Display images
ax1.imshow(img_mechanism)
ax1.axis('off')
ax1.set_title('Alamarblue Redox Dynamics: Michaelis-Menten Kinetics in AMMPER', 
              fontsize=14, fontweight='bold', pad=10)

ax2.imshow(img_wt)
ax2.axis('off')

ax3.imshow(img_rad51)
ax3.axis('off')

# Add panel labels (A, B, C) - aligned on left edge
ax1.text(-0.26, 1.00, 'A', transform=ax1.transAxes, 
         fontsize=20, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=2))

ax2.text(0.01, 1.00, 'B', transform=ax2.transAxes, 
         fontsize=20, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=2))

ax3.text(-0.02, 0.98, 'C', transform=ax3.transAxes, 
         fontsize=20, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=2))

# Adjust layout
plt.tight_layout()

# Create output directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Save in multiple formats
output_base = P.figures('comprehensive_alamarblue_stacked')

plt.savefig(f'{output_base}.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{output_base}.png', format='png', bbox_inches='tight', dpi=300)
plt.savefig(f'{output_base}.svg', format='svg', bbox_inches='tight')

print("Stacked figure created successfully!")
print(f"\nFiles saved:")
print(f"  - {output_base}.pdf (vector)")
print(f"  - {output_base}.png (high-res raster)")
print(f"  - {output_base}.svg (vector, editable)")
print("\nFigure structure:")
print("  Panel A: Mechanism schematic")
print("  Panel B: Wild Type strain (all 6 doses)")
print("  Panel C: rad51Δ strain (all 6 doses)")

"""
Comprehensive Panel Figure: Stack Growth Curves, ROS Ratios, and Half-Life
Creates a vertically stacked publication-ready figure with side-by-side comparisons
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import ammper_paths as P  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os

# ===== UPDATE THESE PATHS =====
# Path to your PNG images
ASSETS = _os.path.join(P.RESULTS, "figures_updated_figures_branch")
GROWTH_CURVES_IMAGE = _os.path.join(ASSETS, "growth_curves_panel.png")
ROS_RATIOS_RAD51_IMAGE = _os.path.join(ASSETS, "rad51UHcellratios.png")
ROS_RATIOS_WT_IMAGE = _os.path.join(ASSETS, "WTUHcellratios.png")
HALF_LIFE_RAD51_IMAGE = _os.path.join(ASSETS, "Halflifeanalysis.png")
HALF_LIFE_WT_IMAGE = _os.path.join(ASSETS, "WTHalflives.png")
# ==============================

def check_files(file_list):
    """Check if all files exist"""
    missing = []
    for filepath in file_list:
        if not os.path.exists(filepath):
            missing.append(filepath)
    return missing

# File list
files_to_check = [
    GROWTH_CURVES_IMAGE,
    ROS_RATIOS_RAD51_IMAGE,
    ROS_RATIOS_WT_IMAGE,
    HALF_LIFE_RAD51_IMAGE,
    HALF_LIFE_WT_IMAGE
]

# Check if files exist
missing_files = check_files(files_to_check)
if missing_files:
    print("ERROR: The following files were not found:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nPlease update the file paths at the top of the script")
    exit(1)

# Load images
img_growth = mpimg.imread(GROWTH_CURVES_IMAGE)
img_ros_rad51 = mpimg.imread(ROS_RATIOS_RAD51_IMAGE)
img_ros_wt = mpimg.imread(ROS_RATIOS_WT_IMAGE)
img_halflife_rad51 = mpimg.imread(HALF_LIFE_RAD51_IMAGE)
img_halflife_wt = mpimg.imread(HALF_LIFE_WT_IMAGE)

# Create figure with custom gridspec layout
fig = plt.figure(figsize=(20, 24))

# Create gridspec: 3 rows
# Row 0: Growth curves (full width)
# Row 1: ROS ratios (2 columns: rad51 | WT)
# Row 2: Half-life (2 columns: rad51 | WT)
gs = GridSpec(3, 2, figure=fig, 
              height_ratios=[1.5, 1, 1],
              width_ratios=[1, 1],
              hspace=0.12,
              wspace=0.08)

# Create subplots
ax1 = fig.add_subplot(gs[0, :])  # Growth curves - spans both columns
ax2 = fig.add_subplot(gs[1, 0])  # rad51 ROS - left
ax3 = fig.add_subplot(gs[1, 1])  # WT ROS - right
ax4 = fig.add_subplot(gs[2, 0])  # rad51 Half-life - left
ax5 = fig.add_subplot(gs[2, 1])  # WT Half-life - right

# Display images
ax1.imshow(img_growth)
ax1.axis('off')

ax2.imshow(img_ros_rad51)
ax2.axis('off')

ax3.imshow(img_ros_wt)
ax3.axis('off')

ax4.imshow(img_halflife_rad51)
ax4.axis('off')

ax5.imshow(img_halflife_wt)
ax5.axis('off')

# Add panel labels (A, B, C, D, E)
label_props = dict(
    fontsize=24, 
    fontweight='bold', 
    va='top', 
    ha='left',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
              edgecolor='black', linewidth=2)
)

ax1.text(-0.02, 0.98, 'A', transform=ax1.transAxes, **label_props)
ax2.text(-0.04, 0.98, 'B', transform=ax2.transAxes, **label_props)
ax3.text(-0.04, 0.98, 'C', transform=ax3.transAxes, **label_props)
ax4.text(-0.04, 0.98, 'D', transform=ax4.transAxes, **label_props)
ax5.text(-0.04, 0.98, 'E', transform=ax5.transAxes, **label_props)

# Add section titles on the right side
title_props = dict(
    fontsize=13,
    fontweight='bold',
    rotation=270,
    va='center',
    ha='left'
)

fig.text(0.985, 0.73, 'Cell Growth Dynamics', **title_props)
fig.text(0.985, 0.43, 'Damage Response Ratios', **title_props)
fig.text(0.985, 0.18, 'DNA Repair Kinetics', **title_props)

# Add strain labels between the columns
strain_props = dict(
    fontsize=12,
    fontweight='bold',
    ha='center',
    va='center',
    style='italic'
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.98, 1])

# Create output directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Save in multiple formats
output_base = P.figures('comprehensive_stacked_panel')

plt.savefig(f'{output_base}.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{output_base}.png', format='png', bbox_inches='tight', dpi=300)
plt.savefig(f'{output_base}.svg', format='svg', bbox_inches='tight')

print("Comprehensive stacked panel created successfully!")
print(f"\nFiles saved:")
print(f"  - {output_base}.pdf (vector)")
print(f"  - {output_base}.png (high-res raster)")
print(f"  - {output_base}.svg (vector, editable)")
print("\nPanel structure:")
print("  Panel A: Cell Growth Dynamics (Healthy vs Unhealthy) - Full Width")
print("  Panel B: rad51Δ Damage Response (Unhealthy/Healthy Ratios) - Left")
print("  Panel C: Wild Type Damage Response (Unhealthy/Healthy Ratios) - Right")
print("  Panel D: rad51Δ DNA Repair Half-Life Analysis - Left")
print("  Panel E: Wild Type DNA Repair Half-Life Analysis - Right")

# plt.show()
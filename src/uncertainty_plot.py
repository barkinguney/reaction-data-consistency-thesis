import pandas as pd
import matplotlib.pyplot as plt


experiment_name = "shen_2009"  # Change this to switch datasets
# Load your CSV file
data = pd.read_csv("data/" + experiment_name + ".csv")

# Make the figure
plt.figure(figsize=(6, 4.5))

# Plot data with error bars
plt.errorbar(
    1000/data["T5_K"], data["ignition_delay_time"], yerr=data["final_extended"],
    fmt="s", markersize=5, color="black",
    mfc="black",   # filled marker
    mec="black",   # marker edge color
    elinewidth=1, capsize=3, label=experiment_name
)

experiment2_name = "davidson_2005"  # Change this to switch datasets
# Example: second dataset (if you have another CSV)
data2 = pd.read_csv("data/" + experiment2_name + ".csv")
plt.errorbar(
    1000/data2["T5_K"], data2["ignition_delay_time"], yerr=data2["final_extended"],
    fmt="s", markersize=5, color="black",
    mfc="white", mec="black",  # open marker
    elinewidth=1, capsize=3, label=experiment2_name
)

# Axis labels
plt.xlabel("1000/T (K)", fontsize=12)
plt.ylabel("Ignition delay time (μs)", fontsize=12)

# Axis limits (adjust if needed)
# plt.xlim(0.9, 1.6)
# plt.ylim(0, 4500)

# # Inset text (top-left corner)
# plt.text(0.72, 1450, "Toluene/air", fontsize=11)
# plt.text(0.72, 1380, "ϕ = 1.0", fontsize=11)
# plt.text(0.72, 1310, "P = 50 atm", fontsize=11)

# Set logarithmic scale on y-axis
# plt.yscale("log")

# Legend
plt.legend(frameon=False, loc="upper right", fontsize=10)

# Clean plot style
plt.grid(False)
plt.tight_layout()

# Save to file
#plt.savefig("plots/" + experiment_name + "_" + experiment2_name + "_ignition_delay.png", dpi=300)
plt.savefig("plots/" + experiment_name + "_ignition_delay.png", dpi=300)
#plt.savefig("plots/" + experiment_name + "_" + experiment2_name + "_ignition_delay.pdf")  # vector format for publications
plt.show()

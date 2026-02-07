from battery_engine import BatteryEngine
import matplotlib.pyplot as plt
import os

# Create image directory
os.makedirs("images", exist_ok=True)
engine = BatteryEngine()

scenarios = [
    {'Label': 'Standby', 'T_amb': 15, 'SOC_init': 0.20, 'N_cycles': 100, 'P_avg': 1.29, 'Color': 'green'},
    {'Label': 'Light Office', 'T_amb': 25, 'SOC_init': 0.55, 'N_cycles': 150, 'P_avg': 2.29, 'Color': 'blue'},
    {'Label': 'Heavy Gaming', 'T_amb': 40, 'SOC_init': 0.85, 'N_cycles': 200, 'P_avg': 6.58, 'Color': 'red'},
    {'Label': 'Extreme Cold', 'T_amb': -10, 'SOC_init': 0.30, 'N_cycles': 250, 'P_avg': 1.18, 'Color': 'cyan'}
]

# Run Simulations
results = [engine.run_tte_sim(s) for s in scenarios]

# Figure 1: SOC Trajectories
plt.figure(figsize=(10, 6))
for i, res in enumerate(results):
    plt.plot(res['time_h'], res['SOC'], label=f"{scenarios[i]['Label']} (TTE: {res['TTE']:.2f}h)",
             color=scenarios[i]['Color'], lw=2.5)

plt.title('Figure 6.2: Comparative SOC Trajectories across Scenarios', fontsize=14)
plt.xlabel('Time (Hours)'); plt.ylabel('SOC (%)'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig("images/SOC_Comparison.png", dpi=300)

# Figure 2: Detailed Subplots (Voltage & SOC)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, res in enumerate(results):
    ax = axes[i//2, i%2]
    ax_soc = ax.twinx()
    ax.plot(res['time_h'], res['V'], color=scenarios[i]['Color'], label='Voltage')
    ax_soc.plot(res['time_h'], res['SOC'], color='gray', linestyle='--')
    ax.set_title(f"{scenarios[i]['Label']} (TTE: {res['TTE']:.2f}h)")
    ax.axhline(3.0, color='r', ls=':')
plt.tight_layout()
plt.savefig("images/TTE_Details.png", dpi=300)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns
import os

# --- 0. Configuration ---
SAVE_DIR = "images"
DPI = 300
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# --- 1. Parameter Management ---
class BatteryParams:
    def __init__(self):
        # Physical Constants
        self.R_gas = 8.314
        self.T_ref = 298.15

        # Kinetics
        self.c = 0.94242
        self.A_k = 7.0e-6
        self.Ea_k = 3500.0

        # Capacity
        self.Q_nom_Ah = 5.000
        self.Q_nom_C = self.Q_nom_Ah * 3600

        # ECM Parameters
        self.R0_ref = 0.0220
        self.Ea_R = 3500.0
        self.R_ts = 0.0450
        self.C_ts = 1781.2
        self.R_tl = 0.1520
        self.C_tl = 10000.0


        self.beta_sei = 0.00730
        self.gamma = 0.00010

        # Thresholds
        self.V_cutoff = 3.0
        self.eta_pmic = 0.90


        self.ocv_coeffs = [1.6271, -0.0468, 1.4111, 3.1296, 5.3748, 3.3392]

    def get_k(self, T_kelvin):
        return self.A_k * np.exp(-self.Ea_k / (self.R_gas * T_kelvin))

    def get_SOH(self, N):
        """Message (SOH) - Message"""
        return 1.0 - self.beta_sei * np.sqrt(N)

    def get_R0(self, T_kelvin, N):
        """Message - Message(Arrhenius)Message"""
        R_therm = self.R0_ref * np.exp(self.Ea_R / (self.R_gas * T_kelvin) - self.Ea_R / (self.R_gas * self.T_ref))
        R_aging = (1.0 + self.gamma * np.sqrt(N))
        return R_therm * R_aging

    def get_capacity_correction(self, T_celsius):
        """CF(T) Message"""
        dT = T_celsius - 25.0
        if T_celsius < 10:
            return -20.01e-6 * dT ** 3 - 1.8511e-3 * dT ** 2 - 41.536e-3 * dT + 0.6559
        elif T_celsius < 25:
            return 5.0424e-8 * dT ** 3 - 3.4652e-4 * dT ** 2 - 3.9226e-3 * dT + 0.9693
        elif T_celsius < 32.5:
            return 9.1472e-6 * dT ** 3 + 6.2834e-5 * dT ** 2 + 2.2177e-3 * dT + 1.0000
        elif T_celsius < 45:
            return -19.362e-6 * dT ** 3 + 2.7307e-4 * dT ** 2 + 1.2833e-3 * dT + 1.0000
        else:
            return 5.8278e-5 * dT ** 3 - 4.5955e-3 * dT ** 2 + 1.0193e-1 * dT + 0.3135

    def get_ocv(self, soc):
        a = self.ocv_coeffs
        term_exp = a[0] * np.exp(a[1] * soc)
        return term_exp + a[2] + a[3] * soc + a[4] * (soc ** 2) + a[5] * (soc ** 3)



def model_derivatives(t, state, params, T_kelvin, Q_eff_C, P_load, R0_val, k_val):
    y1, y2, V_ts, V_tl = state
    h1 = y1 / params.c
    h2 = y2 / (1 - params.c)
    u_t = max(0, (1 - params.c) * (h2 - h1))
    soc_curr = (y1 + y2 - u_t) / Q_eff_C
    soc_curr = np.clip(soc_curr, 0, 1)
    ocv = params.get_ocv(soc_curr)
    V_pol = V_ts + V_tl
    V_est = max(2.5, ocv - V_pol)
    I_load = P_load / (params.eta_pmic * V_est)
    transfer = k_val * (h2 - h1)
    dy1_dt = -I_load + transfer
    dy2_dt = -transfer
    dVts_dt = (I_load / params.C_ts) - (V_ts / (params.R_ts * params.C_ts))
    dVtl_dt = (I_load / params.C_tl) - (V_tl / (params.R_tl * params.C_tl))
    return [dy1_dt, dy2_dt, dVts_dt, dVtl_dt]



def simulate_scenario(setup, params):
    T_c = setup['T_amb']
    soc_init = setup['SOC_init']
    P_avg = setup['P_avg']
    N = setup['N_cycles']
    eta = setup.get('eta', params.eta_pmic)

    T_k = T_c + 273.15
    k_val = params.get_k(T_k)
    R0_val = params.get_R0(T_k, N)
    soh_val = params.get_SOH(N)
    cf_t = params.get_capacity_correction(T_c)


    Q_eff = params.Q_nom_C * soh_val * cf_t

    q_init = Q_eff * soc_init
    y1_0 = q_init * params.c
    y2_0 = q_init * (1 - params.c)
    state0 = [y1_0, y2_0, 0.0, 0.0]

    def cutoff_event(t, state):
        y1, y2, Vts, Vtl = state
        h1 = y1 / params.c
        h2 = y2 / (1 - params.c)
        soc = (y1 + y2 - (1 - params.c) * (h2 - h1)) / Q_eff
        ocv = params.get_ocv(max(0, soc))
        I_load = P_avg / (eta * max(2.5, ocv - (Vts + Vtl)))
        V_term = ocv - I_load * R0_val - (Vts + Vtl)
        return V_term - params.V_cutoff

    cutoff_event.terminal = True;
    cutoff_event.direction = -1

    sol = solve_ivp(
        fun=lambda t, y: model_derivatives(t, y, params, T_k, Q_eff, P_avg, R0_val, k_val),
        t_span=(0, 48 * 3600), y0=state0, events=cutoff_event, method='RK45', rtol=1e-5
    )


    t = sol.t
    y1, y2, Vts, Vtl = sol.y
    h1 = y1 / params.c;
    h2 = y2 / (1 - params.c)
    soc_s = (y1 + y2 - np.maximum(0, (1 - params.c) * (h2 - h1))) / Q_eff
    V_term_s = params.get_ocv(soc_s) - (P_avg / (eta * 3.7)) * R0_val - (Vts + Vtl)

    return {'time_h': t / 3600, 'V': V_term_s, 'SOC': soc_s * 100, 'TTE': t[-1] / 3600}


# --- 4. Main & Scenarios ---
def main():
    params = BatteryParams()
    scenarios = [
        {'Label': 'Standby', 'T_amb': 15, 'SOC_init': 0.20, 'N_cycles': 100, 'P_avg': 1.29, 'Color': 'green',
         'eta': 0.942},
        {'Label': 'Light Office', 'T_amb': 25, 'SOC_init': 0.55, 'N_cycles': 150, 'P_avg': 2.29, 'Color': 'blue',
         'eta': 0.964},
        {'Label': 'Heavy Gaming', 'T_amb': 40, 'SOC_init': 0.85, 'N_cycles': 200, 'P_avg': 6.58, 'Color': 'red',
         'eta': 0.917},
        {'Label': 'Extreme Cold', 'T_amb': -10, 'SOC_init': 0.30, 'N_cycles': 250, 'P_avg': 1.18, 'Color': 'cyan',
         'eta': 0.931}
    ]

    results = []
    for scen in scenarios:
        res = simulate_scenario(scen, params)
        res['Label'] = scen['Label']
        res['Color'] = scen['Color']
        results.append(res)


    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Figure 6.1: Detailed Battery Dynamics across Scenarios', fontsize=16, fontweight='bold')

    for i, res in enumerate(results):
        ax = axes[i // 2, i % 2]
        ax_soc = ax.twinx()


        ax.plot(res['time_h'], res['V'], color=res['Color'], linewidth=2.5, label='Terminal Voltage')

        ax_soc.plot(res['time_h'], res['SOC'], color='gray', linestyle='--', alpha=0.8, label='SOC (%)')

        ax.set_title(f"Scenario: {res['Label']}\nTTE = {res['TTE']:.2f} h", fontsize=13)
        ax.set_xlabel('Time (Hours)')
        ax.set_ylabel('Voltage (V)', color=res['Color'])
        ax_soc.set_ylabel('SOC (%)', color='gray')
        ax.axhline(3.0, color='red', linestyle=':', label='Cutoff')
        ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(SAVE_DIR, "Scenario_Details.png"), dpi=DPI)


    plt.figure(figsize=(12, 7))
    for res in results:
        plt.plot(res['time_h'], res['SOC'], label=f"{res['Label']} (TTE={res['TTE']:.2f}h)",
                 color=res['Color'], linewidth=3)

    plt.title('Figure 6.2: Comparative SOC Trajectories (Predicting Time-to-Empty)', fontsize=15, fontweight='bold')
    plt.xlabel('Discharge Time (Hours)', fontsize=12)
    plt.ylabel('Available State of Charge (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.xlim(0, 5)
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)


    plt.annotate('Fastest Drain (Heavy Gaming)', xy=(1.5, 40), xytext=(2.5, 60),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "SOC_Comparison_Total.png"), dpi=DPI)

    print("Optimization Complete. SOC Comparison graph has been generated.")


if __name__ == "__main__":
    main()
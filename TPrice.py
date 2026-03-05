# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 17:22:11 2025

@author: juanf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

#%% READING EXCEL
df = pd.read_csv( 'MSF568_2025cFall_AnalyticGroupFinalAssignment.csv', index_col=0)

T_mean=df['T_mean'].to_numpy()
T_std=df['T_std'].to_numpy()
L_mean=df['L_mean'].to_numpy()
L_std=df['L_std'].to_numpy()

#%% APPENDIX A --> Creating Td, Ld, Wd

# Dimension's data
t_intervals = 92        # days of simulation btw 1st of July to 30 of September
iterations  = 100000    # number of simulated paths

# Correlation
rhoT  = 0.74
rhoL  = 0.83
rhoTL = 0.88
rhoTW = 0.63
rhoLW = 0.72

dt = 1/365  # time step in years


U1 = np.random.rand(t_intervals, iterations)
U2 = np.random.rand(t_intervals, iterations)
U3 = np.random.rand(t_intervals, iterations)
Z1 = norm.ppf(U1)  
Z2 = norm.ppf(U2)
Z3 = norm.ppf(U3)


rho_12 = rhoTL  # corr(T, L)
rho_13 = rhoTW  # corr(T, W)
rho_23 = rhoLW  # corr(L, W)

L21 = rho_12
L22 = np.sqrt(1 - rho_12**2)

L31 = rho_13
L32 = (rho_23 - rho_12*rho_13) / np.sqrt(1 - rho_12**2)
L33 = np.sqrt(1 - L31**2 - L32**2)

eT = Z1
eL = L21*Z1 + L22*Z2
eW = L31*Z1 + L32*Z2 + L33*Z3


# Calculate Wd initial price
SJUN1   = 61.98   # Initial price on June 1
mu      = 0.03    # Annualized continuously compounded rate of return
sigma   = 0.21    # Annualized volatility

Z_jun30 = norm.ppf(np.random.rand(iterations))
SJUN30  = SJUN1 * np.exp((mu - 0.5*sigma**2)*(29/365) + sigma*np.sqrt(29/365)*Z_jun30)


#Calculate Td, Ld, Wd

Td     = np.zeros((t_intervals+1, iterations))
Ld     = np.zeros((t_intervals+1, iterations))
Wd     = np.zeros((t_intervals+1, iterations))
Td_p_p = np.zeros((t_intervals+1, iterations))
Ld_p_p = np.zeros((t_intervals+1, iterations))

Td_p_p[0, :] = 0.0
Ld_p_p[0, :] = 0.0
Wd[0, :]     = SJUN30  

for t in range(t_intervals):
    Td_p_p[t+1, :] = rhoT*Td_p_p[t, :] + np.sqrt(1 - rhoT**2)*eT[t, :]
    Ld_p_p[t+1, :] = rhoL*Ld_p_p[t, :] + np.sqrt(1 - rhoL**2)*eL[t, :]

    Td[t, :] = T_mean[t] + T_std[t]*Td_p_p[t+1, :]
    Ld[t, :] = L_mean[t] + L_std[t]*Ld_p_p[t+1, :]
    
    Wd[t+1, :] = Wd[t, :] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*eW[t, :])

    

#%% MC paths + final histograms for T, L and W (sharper MC lines)

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

T_paths = Td[:t_intervals, :]
L_paths = Ld[:t_intervals, :]
W_paths = Wd[1:t_intervals+1, :]

dates = pd.date_range(start='2025-07-01', periods=t_intervals, freq='D')

# Fewer paths, clearer lines
n_plot = min(150, iterations)
plot_idx = np.linspace(0, iterations - 1, n_plot, dtype=int)

def kfmt(x, pos):
    return f"{x:.0f}"

# ===== 1) Temperature =====
fig_T, (ax_T_paths, ax_T_hist) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=False
)

for j in plot_idx:
    ax_T_paths.plot(dates, T_paths[:, j],
                    color='tab:blue', alpha=0.15, linewidth=1.0)  # <- más nítido

ax_T_paths.plot(dates, T_paths.mean(axis=1),
                color='navy', linewidth=2.2, label='Mean path')

ax_T_paths.set_title('Simulated Temperature Paths (Td)')
ax_T_paths.set_ylabel('Temperature (°F)')
ax_T_paths.grid(True, which='both', linestyle=':', linewidth=0.7)
ax_T_paths.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax_T_paths.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
fig_T.autofmt_xdate()
ax_T_paths.legend(loc='upper left')

final_T = T_paths[-1, :]
ax_T_hist.hist(final_T, bins=50, density=True,
               color='tab:blue', alpha=0.7, edgecolor='black')
ax_T_hist.set_xlabel('Final temperature (°F)')
ax_T_hist.set_ylabel('Density')
ax_T_hist.grid(True, linestyle=':', linewidth=0.7)
ax_T_hist.xaxis.set_major_formatter(FuncFormatter(kfmt))

plt.tight_layout()
plt.show()

# ===== 2) Load =====
fig_L, (ax_L_paths, ax_L_hist) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=False
)

for j in plot_idx:
    ax_L_paths.plot(dates, L_paths[:, j],
                    color='tab:green', alpha=0.15, linewidth=1.0)

ax_L_paths.plot(dates, L_paths.mean(axis=1),
                color='darkgreen', linewidth=2.2, label='Mean path')

ax_L_paths.set_title('Simulated Load Paths (Ld)')
ax_L_paths.set_ylabel('Load (MW)')
ax_L_paths.grid(True, which='both', linestyle=':', linewidth=0.7)
ax_L_paths.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax_L_paths.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
fig_L.autofmt_xdate()
ax_L_paths.legend(loc='upper left')

final_L = L_paths[-1, :]
ax_L_hist.hist(final_L, bins=50, density=True,
               color='tab:green', alpha=0.7, edgecolor='black')
ax_L_hist.set_xlabel('Final load (MW)')
ax_L_hist.set_ylabel('Density')
ax_L_hist.grid(True, linestyle=':', linewidth=0.7)
ax_L_hist.xaxis.set_major_formatter(FuncFormatter(kfmt))

plt.tight_layout()
plt.show()

# ===== 3) Spot price =====
fig_W, (ax_W_paths, ax_W_hist) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=False
)

for j in plot_idx:
    ax_W_paths.plot(dates, W_paths[:, j],
                    color='tab:orange', alpha=0.15, linewidth=1.0)

ax_W_paths.plot(dates, W_paths.mean(axis=1),
                color='darkorange', linewidth=2.2, label='Mean path')

ax_W_paths.set_title('Simulated Spot Price Paths (Wd)')
ax_W_paths.set_ylabel('Spot price (USD/MWh)')
ax_W_paths.grid(True, which='both', linestyle=':', linewidth=0.7)
ax_W_paths.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax_W_paths.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
fig_W.autofmt_xdate()
ax_W_paths.legend(loc='upper left')

final_W = W_paths[-1, :]
ax_W_hist.hist(final_W, bins=50, density=True,
               color='tab:orange', alpha=0.7, edgecolor='black')
ax_W_hist.set_xlabel('Final spot price (USD/MWh)')
ax_W_hist.set_ylabel('Density')
ax_W_hist.grid(True, linestyle=':', linewidth=0.7)
ax_W_hist.xaxis.set_major_formatter(FuncFormatter(kfmt))

plt.tight_layout()
plt.show()


#%% Simulate PI a)
R = 70.48
pi = np.zeros(iterations)   

for j in range(iterations):      
    s = 0.0
    for i in range(t_intervals): 
        s += Ld[i, j] * (R - Wd[i+1, j])
    pi[j] = s


# Visualization of Unhedged Profit (Part A) ---

plt.figure(figsize=(12, 7))

# 1. Plot the Histogram
# We use 'bins=100' for good granularity. 'alpha=0.6' makes it slightly transparent.
plt.hist(pi, bins=100, color='#607c8e', edgecolor='white', alpha=0.7, label='Profit Frequency')

# 2. Calculate Statistics
mean_pi = np.mean(pi)
var_95_pi = np.percentile(pi, 5)
var_99_pi = np.percentile(pi, 1)

# 3. Add Vertical Lines for Mean, 5%, and 1%
plt.axvline(mean_pi, color='green', linestyle='--', linewidth=2.5, 
            label=f'Mean Expected: ${mean_pi:,.0f}')

plt.axvline(var_95_pi, color='orange', linestyle='--', linewidth=2.5, 
            label=f'5% VaR (95%): ${var_95_pi:,.0f}')

plt.axvline(var_99_pi, color='red', linestyle='--', linewidth=2.5, 
            label=f'1% VaR (99%): ${var_99_pi:,.0f}')

# 4. Add Text Annotations (Optional, for clarity)
# Places text slightly above the max frequency to make it readable
y_max = plt.ylim()[1] 
plt.text(var_99_pi, y_max*0.5, f' 1% Worst Case\n ${var_99_pi/1e6:.1f}M', color='red', ha='right')

# 5. Formatting
plt.title('Distribution of Unhedged Summer Revenue (Part A)', fontsize=16, fontweight='bold')
plt.xlabel('Total Revenue ($)', fontsize=12)
plt.ylabel('Frequency (Number of Scenarios)', fontsize=12)
plt.legend(loc='upper left', frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()

# Statictics
print ('Pi (Part A):')
print ( "mean(pi)=","{:,.0f}".format(mean_pi) )
print ( "median(pi)=","{:,.0f}".format(np.median(pi)) )
print ( "5 percentile (pi)=","{:,.0f}".format(var_95_pi) )
print ( "1 percentile (pi)=","{:,.0f}".format(var_99_pi) )

#%% Hedging with electricity forward only b)
F_elec = 67.0 
Vw = 10

#Calculating payoffs
sumWd = np.zeros(iterations)
payoff_elec = np.zeros(iterations)

for j in range(iterations):
    s = 0.0
    for i in range(t_intervals):
        s += Wd[i+1, j]
    sumWd[j] = s   

payoff_elec = 92*24*Vw*((sumWd/92)-F_elec)

#Looking for the best number of contracts
pi_hedgedE = np.zeros(iterations)
n_valuesE = np.arange(-50, 51)   
p1_valuesE = []                

for n in n_valuesE:
    pi_hedgedE = pi + n * payoff_elec
    p1 = np.percentile(pi_hedgedE, 1)
    p1_valuesE.append(p1)

p1_valuesE = np.array(p1_valuesE)
best_idxE = np.argmax(p1_valuesE)
best_nE = n_valuesE[best_idxE]
best_p1E = p1_valuesE[best_idxE]

print("\nHedging with electricity forwards (Part b)")
print("Best number of contracts (n) =", best_nE)
print("Max 1%-percentile of hedged profit =", "{:,.0f}".format(best_p1E))

# Si quieres ver también mean/median para ese n óptimo:
pi_bestE = pi + best_nE * payoff_elec
print("mean(pi_hedged)   =", "{:,.0f}".format(np.mean(pi_bestE)))
print("median(pi_hedged) =", "{:,.0f}".format(np.median(pi_bestE)))
print("5%-percentile     =", "{:,.0f}".format(np.percentile(pi_bestE, 5)))
print("1%-percentile     =", "{:,.0f}".format(np.percentile(pi_bestE, 1)))

#%% Hedging with temperature forward only c)
F_temp =  684
Vh = 20

#Calculating payoffs
sumMax = np.zeros(iterations)
payoff_temp = np.zeros(iterations)

for j in range(iterations):
    s = 0.0
    for i in range(t_intervals):
        s += max(Td[i, j]-65,0)
    sumMax[j] = s   

payoff_temp = Vh*((sumMax)-F_temp)

#Looking for the best number of contracts
pi_hedgedT = np.zeros(iterations)
n_valuesT = np.arange(-50,1000 )   
p1_valuesT = []                

for n in n_valuesT:
    pi_hedgedT = pi + n * payoff_temp
    p1 = np.percentile(pi_hedgedT, 1)
    p1_valuesT.append(p1)

p1_valuesT = np.array(p1_valuesT)
best_idxT = np.argmax(p1_valuesT)
best_nT = n_valuesT[best_idxT]
best_p1T = p1_valuesT[best_idxT]

print("\nHedging with temperature forwards (Part c)")
print("Best number of contracts (n) =", best_nT)
print("Max 1%-percentile of hedged profit =", "{:,.0f}".format(best_p1T))

# Si quieres ver también mean/median para ese n óptimo:
pi_bestT = pi + best_nT * payoff_temp
print("mean(pi_hedged)   =", "{:,.0f}".format(np.mean(pi_bestT)))
print("median(pi_hedged) =", "{:,.0f}".format(np.median(pi_bestT)))
print("5%-percentile     =", "{:,.0f}".format(np.percentile(pi_bestT, 5)))
print("1%-percentile     =", "{:,.0f}".format(np.percentile(pi_bestT, 1)))


#%% Linear / Integer Programming hedging model (electricity + CDD forwards) fixed theta
import pulp as pl
import numpy as np

# -------------------------------------------------
# 1. Basic parameters from simulation
# -------------------------------------------------
N = iterations            # number of MC scenarios

# Mean payoffs per contract
Pbar_E = payoff_elec.mean()
Pbar_T = payoff_temp.mean()

# Expected profit without hedge
mu0 = pi.mean()

# -------------------------------------------------
# 2. Physical exposure in electricity
# -------------------------------------------------
# Energy per electricity forward (MWh)
Q_contrato = t_intervals * 24 * Vw   # 92 days * 24h * 10 MW

# Expected physical energy sold in summer (MWh)
# Sum over days 0..(t_intervals-1), then average over scenarios
Q_elec = Ld[:t_intervals, :].sum(axis=0).mean()

# Max hedge factor (e.g. 120% of physical exposure)
kappa_E = 1.2

# Continuous max number of contracts
nE_max_cont = kappa_E * Q_elec / Q_contrato

# Integer max number of contracts (floor)
nE_max = int(np.floor(nE_max_cont))

print("Q_contrato (MWh) =", Q_contrato)
print("Q_elec (MWh, expected) =", Q_elec)
print("nE_max (integer) =", nE_max)

# -------------------------------------------------
# 3. Tail (stress) scenarios
# -------------------------------------------------
alpha = 0.01          # tail level (1%)
K = int(alpha * N)    # number of tail scenarios

# Indices of the K worst scenarios without hedge
tail_idx = np.argsort(pi)[:K]

# Minimum profit required in tail scenarios
p1_unhedged = np.percentile(pi, 1)   # ~ -789448
theta = 0.45                          # 20% of current loss
B_min = theta * p1_unhedged 

print("Number of stress scenarios (K) =", K)
print("B_min for tail scenarios =", B_min)

# -------------------------------------------------
# 4. MIP model
# -------------------------------------------------
prob = pl.LpProblem("ABC_Hedging_MIP", pl.LpMaximize)

# Integer decision variables: number of contracts
n_E = pl.LpVariable('n_E', lowBound=-nE_max, upBound=nE_max, cat='Integer')
n_T = pl.LpVariable('n_T', lowBound=None,   upBound=None,   cat='Integer')
# Note: you can add your own bounds/constraints for n_T later

# -------------------------------------------------
# 5. Objective function
# -------------------------------------------------
# Maximize expected profit increment:
# E[pi_H] = mu0 + n_E * Pbar_E + n_T * Pbar_T
# mu0 is constant, so we maximize n_E * Pbar_E + n_T * Pbar_T
prob += n_E * Pbar_E + n_T * Pbar_T, "Expected_profit_increment"

# -------------------------------------------------
# 6. Tail constraints
# -------------------------------------------------
# For each stress scenario: pi[idx] + n_E * payoff_elec[idx] + n_T * payoff_temp[idx] >= B_min
for idx in tail_idx:
    prob += (
        pi[idx]
        + n_E * float(payoff_elec[idx])
        + n_T * float(payoff_temp[idx])
        >= B_min
    )

# -------------------------------------------------
# 7. Solve MIP
# -------------------------------------------------
prob.solve(pl.PULP_CBC_CMD(msg=False))

print("\n--- Hedging MIP solution (electricity + CDD) ---")
print("Status:", pl.LpStatus[prob.status])
print("Optimal n_E (electricity forwards) =", n_E.value())
print("Optimal n_T (CDD forwards)         =", n_T.value())

# -------------------------------------------------
# 8. Analyse new profit distribution
# -------------------------------------------------
mu_hedged = mu0 + n_E.value() * Pbar_E + n_T.value() * Pbar_T
print("E[pi] without hedge =", "{:,.0f}".format(mu0))
print("E[pi] with hedge    =", "{:,.0f}".format(mu_hedged))

pi_hedged_opt = pi + n_E.value() * payoff_elec + n_T.value() * payoff_temp
print("1%-percentile pi (no hedge)  =", "{:,.0f}".format(np.percentile(pi, 1)))
print("1%-percentile pi (with hedge)=", "{:,.0f}".format(np.percentile(pi_hedged_opt, 1)))
print("5%-percentile pi (no hedge)  =", "{:,.0f}".format(np.percentile(pi, 5)))
print("5%-percentile pi (with hedge)=", "{:,.0f}".format(np.percentile(pi_hedged_opt, 5)))



#%% Integer Programming hedging model with theta sweep (variable theta)
import pulp as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Basic parameters from simulation
N = iterations                    # number of MC scenarios
Pbar_E = payoff_elec.mean()       # mean payoff electricity forward
Pbar_T = payoff_temp.mean()       # mean payoff CDD forward
mu0 = pi.mean()                   # expected profit without hedge

# 2. Physical exposure in electricity
Q_contrato = t_intervals * 24 * Vw        # MWh per electricity forward
Q_elec = Ld[:t_intervals, :].sum(axis=0).mean()  # expected MWh sold in summer

kappa_E = 1.2                              # max hedge factor
nE_max_cont = kappa_E * Q_elec / Q_contrato
nE_max = int(np.floor(nE_max_cont))        # integer max number of contracts

print("Q_contrato (MWh) =", Q_contrato)
print("Q_elec (MWh, expected) =", Q_elec)
print("nE_max (integer) =", nE_max)

# 3. Tail (stress) scenarios
alpha = 0.01                  # tail level (1%)
K = int(alpha * N)            # number of tail scenarios
tail_idx = np.argsort(pi)[:K] # indices of worst scenarios

# 1% percentile without hedge
p1_unhedged = np.percentile(pi, 1)
print("1%-percentile pi (no hedge) =", "{:,.0f}".format(p1_unhedged))

# 4. Theta sweep
theta_values = np.arange(0.0, 1.01, 0.05)
results = []

for theta in theta_values:
    # B_min based on theta
    B_min = theta * p1_unhedged

    # Define MIP model
    prob = pl.LpProblem(f"ABC_Hedging_MIP_theta_{theta:.2f}", pl.LpMaximize)

    # Integer decision variables
    n_E = pl.LpVariable('n_E', lowBound=-nE_max, upBound=nE_max, cat='Integer')
    n_T = pl.LpVariable('n_T', lowBound=None,   upBound=None,   cat='Integer')

    # Objective: maximize expected profit increment
    prob += n_E * Pbar_E + n_T * Pbar_T, "Expected_profit_increment"

    # Tail constraints: profit in worst scenarios must be >= B_min
    for idx in tail_idx:
        prob += (
            pi[idx]
            + n_E * float(payoff_elec[idx])
            + n_T * float(payoff_temp[idx])
            >= B_min
        )

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    status = pl.LpStatus[prob.status]
    if status != 'Optimal':
        print(f"Theta {theta:.2f}: status = {status}, skipping")
        continue

    nE_opt = n_E.value()
    nT_opt = n_T.value()

    # Profit distribution with hedge
    pi_hedged = pi + nE_opt * payoff_elec + nT_opt * payoff_temp
    mu_hedged = pi_hedged.mean()
    p1_hedged = np.percentile(pi_hedged, 1)

    print(f"Theta {theta:.2f}: n_E={nE_opt}, n_T={nT_opt}, "
          f"mean_pi_hedged={mu_hedged:,.0f}, p1_hedged={p1_hedged:,.0f}")

    results.append({
        "theta": theta,
        "n_E": nE_opt,
        "n_T": nT_opt,
        "mean_pi_hedged": mu_hedged,
        "p1_pi_hedged": p1_hedged
    })

# 5. Results table
results_df = pd.DataFrame(results)



#%%

import plotly.graph_objects as go

# results_df must contain: "theta", "mean_pi_hedged", "p1_pi_hedged"

fig = go.Figure()

# 3D line + markers
fig.add_trace(go.Scatter3d(
    x=results_df["theta"],               # theta
    y=results_df["p1_pi_hedged"],       # 1% percentile
    z=results_df["mean_pi_hedged"],     # average pi
    mode='lines+markers',
    line=dict(width=5),
    marker=dict(size=5),
    text=[
        f"θ={row.theta:.2f}<br>"
        f"mean={row.mean_pi_hedged:,.0f}<br>"
        f"p1={row.p1_pi_hedged:,.0f}"
        for _, row in results_df.iterrows()
    ],
    hoverinfo='text',
    name='Theta frontier'
))

fig.update_layout(
    title="<b>Theta frontier: 1% percentile vs Average Profit</b>",
    template='plotly_white',
    scene=dict(
        xaxis=dict(
            title='<b>theta</b>',
            backgroundcolor="white",
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title='<b>1% percentile of π</b>',
            backgroundcolor="white",
            gridcolor="lightgray"
        ),
        zaxis=dict(
            title='<b>Average π</b>',
            backgroundcolor="white",
            gridcolor="lightgray"
        ),
        aspectratio=dict(x=1, y=1, z=0.7),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=False
)

# Optional: save to HTML
fig.write_html("theta_frontier.html")

fig.show()


#%%

from matplotlib.ticker import FuncFormatter

def kfmt(x, pos):
    return f"{int(round(x/1000))}k"

fig, ax1 = plt.subplots(figsize=(10, 6))

# Left y-axis: average profit
color1 = 'tab:blue'
ax1.set_xlabel("theta", fontsize=15)  # axis label size
ax1.set_ylabel("Average profit (USD)", color=color1, labelpad=8, fontsize=14)
line1 = ax1.plot(
    results_df["theta"],
    results_df["mean_pi_hedged"],
    marker='o',
    markersize=5,
    linestyle='-',
    color=color1,
    label="Average profit"
)
ax1.yaxis.set_major_formatter(FuncFormatter(kfmt))
ax1.grid(True, which='both', linestyle='--', alpha=0.4)
ax1.tick_params(axis='both', labelsize=15, labelcolor=color1)  # tick labels

# Right y-axis: 1%-percentile
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel("1%-percentile (USD)", color=color2, labelpad=8, fontsize=15)
line2 = ax2.plot(
    results_df["theta"],
    results_df["p1_pi_hedged"],
    marker='s',
    markersize=5,
    linestyle='--',
    color=color2,
    label="1%-percentile"
)
ax2.yaxis.set_major_formatter(FuncFormatter(kfmt))
ax2.tick_params(axis='y', labelsize=15, labelcolor=color2)

fig.suptitle("Impact of theta on average profit and 1%-percentile",
             fontsize=25)  # title size

lines = line1 + line2
labels = [l.get_label() for l in lines]
fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    frameon=False,
    fontsize=15  # legend text size
)

fig.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.show()

#%% PART 4: Hybrid Hedge Strategy Simulation (Grid Search)

# 1. Setup the Simulation Grid

# We will vary the "Electricity Weight" from 0% to 100%

# We use the optimal n found in (b) and (c) as the "100%" reference.

beta_elec_range = np.linspace(0, 1, 101) # 0.00, 0.01, ... 1.00
results_hybrid = []

print("\n--- Running Hybrid Hedge Simulation ---")

for w_e in beta_elec_range:
    w_t = 1.0 - w_e
    
    # Calculate mix of contracts based on the individual bests found earlier
    n_e_curr = int(np.round(w_e * best_nE))
    n_t_curr = int(np.round(w_t * best_nT))
    
    # Calculate Hedged Profit for this specific mix
    # pi is the unhedged profit (from Part A)
    pi_hybrid = pi + (n_e_curr * payoff_elec) + (n_t_curr * payoff_temp)
    
    # Collect Metrics
    stats = {
        'Beta_Elec': w_e * 100,
        'Beta_CDD': w_t * 100,
        'n_E': n_e_curr,
        'n_T': n_t_curr,
        'Mean_Profit': np.mean(pi_hybrid),
        'P1_Profit': np.percentile(pi_hybrid, 1), # 1st Percentile (VaR 99%)
        'P5_Profit': np.percentile(pi_hybrid, 5)  # 5th Percentile (VaR 95%)
    }
    results_hybrid.append(stats)

df_hybrid = pd.DataFrame(results_hybrid)


# 2. Find the "Best" Hybrid Mix (Max 1% Tail)
best_hybrid_idx = df_hybrid['P1_Profit'].idxmax()
best_hybrid = df_hybrid.loc[best_hybrid_idx]

print("Optimal Hybrid Strategy Found:")
print(f"Mix: {best_hybrid['Beta_Elec']:.0f}% Elec / {best_hybrid['Beta_CDD']:.0f}% CDD")
print(f"Contracts: n_E = {best_hybrid['n_E']:.1f}, n_T = {best_hybrid['n_T']:.1f}")
print(f"Resulting 1% Tail: ${best_hybrid['P1_Profit']:,.0f}")


# 3. Visualization
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 1% Tail Risk (Primary Axis)
color = 'tab:blue'
ax1.set_xlabel('Hedge Allocation (% Electricity Forward)')
ax1.set_ylabel('1% Worst-Case Profit ($)', color=color)
ax1.plot(df_hybrid['Beta_Elec'], df_hybrid['P1_Profit'], color=color, linewidth=2.5, label='1% Tail Risk (Maximize This)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which='both', linestyle=':', linewidth=0.7)

# Plot Expected Mean Profit (Secondary Axis) - To see the cost of hedging
ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel('Expected Mean Profit ($)', color=color)  
ax2.plot(df_hybrid['Beta_Elec'], df_hybrid['Mean_Profit'], color=color, linestyle='--', linewidth=2, label='Expected Profit')
ax2.tick_params(axis='y', labelcolor=color)

# Mark the optimal point
ax1.axvline(x=best_hybrid['Beta_Elec'], color='red', linestyle=':', label=f"Optimal Mix ({best_hybrid['Beta_Elec']:.0f}%)")

# Title and Layout
plt.title('Hybrid Hedge Strategy: Risk vs. Allocation \n(Trade-off between Elec and CDD Contracts)')
fig.tight_layout()
plt.show()

# 4. Optional: Efficient Frontier Plot (Risk vs Return)
plt.figure(figsize=(10, 6))
plt.scatter(df_hybrid['P1_Profit'], df_hybrid['Mean_Profit'], c=df_hybrid['Beta_Elec'], cmap='viridis', s=50)
plt.colorbar(label='% Electricity Allocation')
plt.xlabel('Downside Risk Protection (1% Percentile Profit)')
plt.ylabel('Expected Return (Mean Profit)')
plt.title('Efficient Frontier: Hybrid Hedging')
plt.grid(True, linestyle=':', alpha=0.6)

# Annotate Start (0%), End (100%), and Optimal
plt.annotate('100% CDD', (df_hybrid.iloc[0]['P1_Profit'], df_hybrid.iloc[0]['Mean_Profit']), xytext=(10, -10), textcoords='offset points')
plt.annotate('100% Elec', (df_hybrid.iloc[-1]['P1_Profit'], df_hybrid.iloc[-1]['Mean_Profit']), xytext=(10, 10), textcoords='offset points')
plt.annotate('Optimal', (best_hybrid['P1_Profit'], best_hybrid['Mean_Profit']), xytext=(-40, 20), textcoords='offset points', arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

#%% PART 5: ADVANCED ANALYTICS & GLOBAL SEARCH

#Convergence Check (Technical Robustness)

running_mean = np.cumsum(pi) / np.arange(1, iterations + 1)

plt.figure(figsize=(10, 5))
plt.plot(running_mean, color='purple', linewidth=1.5)
plt.axhline(np.mean(pi), color='black', linestyle='--', label='Final Mean')
plt.title('Monte Carlo Convergence Test\n(Stabilization of Mean Profit)', fontsize=14)
plt.xlabel('Number of Iterations')
plt.ylabel('Running Mean Profit ($)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print("Convergence Check: The plot shows the mean stabilizing, confirming N is sufficient.")
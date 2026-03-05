# TPriceHedging

A comprehensive Monte Carlo simulation and optimization framework for analyzing temperature and price hedging strategies in electricity markets.

## Overview

This project analyzes hedging strategies for summer electricity revenue exposed to temperature and electricity price fluctuations. It implements multiple hedging approaches including:

- **Unhedged scenarios** - baseline profit distribution
- **Electricity forward hedging** - using futures contracts on electricity prices
- **Temperature forward hedging** - using CDD (Cooling Degree Day) contracts to hedge temperature risk
- **Hybrid hedging strategies** - combining both forwards with various allocations
- **Integer Programming optimization** - optimal contract selection using mathematical optimization

## Problem Context

The analysis simulates summer revenue (July 1 - September 30, 92 days) for an electricity seller with exposure to:
- **Temperature risk** - affects load and demand
- **Electricity price risk** - volatility in spot prices
- **Load/demand risk** - correlated with temperature

The simulation uses real market parameters and correlation structures to generate 100,000 Monte Carlo scenarios.

## Key Components

### 1. Monte Carlo Simulation (`TPrice.py`)
- Generates correlated paths for:
  - Temperature (Td) with mean-reversion dynamics
  - Electricity Load (Ld) correlated with temperature
  - Spot Price (Wd) with geometric Brownian motion
  
### 2. Hedging Analysis Modules

#### Part A: Unhedged Profit Distribution
- Calculates base case profit: π = Σ Load_i × (Reference_Price - Spot_Price_i)
- Computes risk metrics: mean, median, VaR at 5% and 1% levels
- Visualizes profit distribution

#### Part B: Electricity Forward Hedging
- Tests 101 different contract quantities (-50 to +50)
- Finds optimal n maximizing 1% worst-case profit
- Analyzes payoff structures

#### Part C: Temperature Forward Hedging
- Uses CDD (Cooling Degree Days) contracts
- CDD = max(Temperature - 65°F, 0)
- Optimizes contract count across wider range (-50 to +1000)

#### Part D: Mixed Integer Programming (MIP) Optimization
- **Fixed theta variant**: Minimizes tail loss at fixed confidence level
- **Variable theta sweep**: Parametric analysis across risk thresholds
- Decision variables: n_E (electricity contracts), n_T (CDD contracts)
- Objective: Maximize expected profit while ensuring tail protection
- Constraints: Worst-case (1%) scenarios must exceed minimum threshold

#### Part E: Hybrid Strategy Analysis
- Grid search across electricity/CDD allocation weights (0-100%)
- Generates efficient frontier: risk vs. expected return
- Identifies optimal blend maximizing 1% tail protection
- Visualizes trade-offs

#### Part F: Convergence Validation
- Monte Carlo convergence test
- Confirms stability of mean profit estimate
- Validates sufficient simulation sample size

## Technical Stack

**Language**: Python

**Key Libraries**:
- `numpy`, `pandas` - numerical computing and data manipulation
- `matplotlib` - static visualizations
- `plotly` - interactive 3D visualizations
- `scipy.stats` - statistical distributions
- `pulp` - integer linear programming solver (CBC backend)

## Input Data

Requires CSV file: `MSF568_2025cFall_AnalyticGroupFinalAssignment.csv`

Contains:
- `T_mean` - daily mean temperatures (°F)
- `T_std` - daily temperature standard deviations
- `L_mean` - daily mean electricity load (MW)
- `L_std` - daily load standard deviations

## Output Artifacts

- **Console output**: Summary statistics and optimization results
- **Matplotlib plots**: 
  - Temperature, load, and price simulation paths
  - Profit distribution histograms
  - Hedging impact analysis charts
  - Efficient frontier visualization
- **Interactive HTML**: `theta_frontier.html` (3D surface of theta parameterization)

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| t_intervals | 92 | Simulation days (July-September) |
| iterations | 100,000 | Monte Carlo scenarios |
| R | $70.48 | Reference electricity price ($/MWh) |
| F_elec | $67.00 | Electricity forward strike price |
| F_temp | 684 CDD | Temperature forward strike |
| Vw | 10 MW | Contract size (electricity) |
| Vh | 20 | Contract size (CDD) |
| rhoTL | 0.88 | Temperature-Load correlation |
| rhoTW | 0.63 | Temperature-Price correlation |
| rhoLW | 0.72 | Load-Price correlation |

## Usage

```bash
python TPrice.py

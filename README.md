# Adaptive Wave Trading Model

A physics-based trading model implementing the adaptive-wave alternative to the Black-Scholes option pricing model, based on nonlinear Schrödinger (NLS) equations and fluid dynamics principles.

**Reference:** [Adaptive–Wave Alternative for the Black–Scholes Option Pricing Model](https://arxiv.org/pdf/0911.1834) (arXiv:0911.1834v1 [q-fin.PR])

## Overview

This implementation provides:

1. **Adaptive Nonlinear Schrödinger (NLS) Equation Model** - Four analytical solutions using Jacobi elliptic functions
2. **Manakov System** - Coupled NLS equations for stochastic volatility modeling
3. **Hebbian Learning** - Adaptive market potential estimation
4. **Trading Engine** - Signal generation and backtesting framework
5. **Visualization Tools** - Comprehensive plotting capabilities

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete simulation suite:

```bash
python simulate.py
```

This will generate:
- Visualizations of NLS solutions
- Manakov system evolution plots
- Trading backtest results
- Performance metrics

## Usage Examples

### Basic NLS Model

```python
from nls_model import NLSModel

# Initialize model
nls = NLSModel(sigma=0.3, beta=0.05, k=1.2)

# Calculate probability density for shock-wave solution
s = np.linspace(0, 200, 1000)
pdf = nls.probability_density('shock_wave', s=s, t=1.0, r=0.05)

# Calculate Greeks
greeks = nls.greeks(s=100, t=1.0, r=0.05)
```

### Manakov System

```python
from manakov_system import ManakovSystem

# Initialize Manakov system
manakov = ManakovSystem(r=0.05, c=0.7, N=10)

# Solve using analytical soliton solution
s_grid = np.linspace(-10, 20, 200)
t_grid = np.linspace(0, 5, 50)
result = manakov.solve(s_grid, t_grid, method='soliton')

# Extract volatility and option price PDFs
volatility_pdf = np.abs(result['sigma'])**2
option_pdf = np.abs(result['psi'])**2
```

### Trading Engine

```python
from trading_engine import WaveTradingEngine
import numpy as np

# Initialize trading engine
engine = WaveTradingEngine(initial_capital=100000, risk_free_rate=0.05)

# Generate trading signal
signal = engine.generate_trading_signal(
    current_price=100.0,
    time_horizon=1.0
)

# Execute trade
trade = engine.execute_trade(signal, position_size=0.2)

# Run backtest
prices = np.random.lognormal(4.6, 0.2, 252)  # Synthetic data
results = engine.backtest(prices, time_horizon=1.0, position_size=0.2)

# Get performance metrics
metrics = engine.get_performance_metrics(results)
```

## Model Components

### NLS Solutions

The model implements four analytical solutions:

1. **Jacobi Sine Solution** (12) - General periodic solution
2. **Shock-Wave Solution** (13) - Dark soliton (tanh)
3. **Jacobi Cosine Solution** (14) - General periodic solution
4. **Soliton Solution** (15) - Bright soliton (sech)

### Manakov System

The coupled system models:
- **Volatility evolution**: `i*∂t*σ = -0.5*∂ss*σ - β(|σ|² + |ψ|²)*σ`
- **Option price evolution**: `i*∂t*ψ = -0.5*∂ss*ψ - β(|σ|² + |ψ|²)*ψ`
- **Hebbian learning**: `ẇi = -wi + c*|σ|*gi*|ψ|`

### Trading Signals

Signals are generated based on:
- Expected price from wave function PDF
- Predicted volatility from Manakov system
- Confidence based on PDF concentration
- Risk-adjusted expected returns

## Visualization

The package includes comprehensive visualization tools:

```python
import visualization as viz

# Plot NLS solutions
viz.plot_nls_solutions(nls_model)

# Plot wave evolution
viz.plot_wave_evolution(nls_model, solution_type='shock_wave')

# Plot Manakov system
viz.plot_manakov_system(manakov)

# Plot trading results
viz.plot_trading_results(results, price_series=prices)

# Plot Greeks
viz.plot_greeks(nls_model)
```

## Mathematical Background

The model is based on the adaptive nonlinear Schrödinger equation:

```
i*∂t*ψ = -0.5*σ*∂ss*ψ - β*|ψ|²*ψ
```

where:
- `ψ(s,t)` is the complex-valued wave function
- `|ψ(s,t)|²` is the probability density function for option price
- `σ` is volatility (dispersion frequency coefficient)
- `β(r,w)` is adaptive market potential

The wave function approach provides a quantum-mechanical perspective on financial markets, where probability amplitudes evolve according to nonlinear wave dynamics.

## Performance Considerations

- Analytical solutions are fast but limited to specific forms
- Numerical solutions of Manakov system can be computationally intensive
- For large-scale backtesting, consider using analytical approximations

## References

1. Ivancevic, V. G. (2009). Adaptive–Wave Alternative for the Black–Scholes Option Pricing Model. arXiv:0911.1834v1 [q-fin.PR]

2. Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637-659.

3. Manakov, S. V. (1973). On the theory of two-dimensional stationary self-focusing of electromagnetic waves. Soviet Physics JETP, 38, 248-253.

## License

This implementation is for educational and research purposes.

## Disclaimer

This model is a theoretical framework based on physics principles. It should not be used for actual trading without thorough validation and risk management. Past performance does not guarantee future results.


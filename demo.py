# Project imports
import core

param_list = [
    {"b": 1.5, "c": 0.5, "d": 0.1, "s": 0.2,
     "Z": 100, "beta": 1, "mu": 1e-3, "strategies": ['CC', 'HD', 'TD']},
]

simulation_fn = None # Currently, no simulation_fn exists
# plotting_fn = core.plot_simplex # ~30s per plot
# plotting_fn = core.plot_simplex_numerical # ~150s per plot
plotting_fn = core.plot_deception_frequency
plotting_fn = core.plot_strategy_frequencies
results = core.run_all_simulations(param_list,
                                   simulation_fn=simulation_fn,
                                   plotting_fn=plotting_fn,
                                   plot_dir="plots",
                                   simulation_dir="dataframes",)

figs, simulation_results = results
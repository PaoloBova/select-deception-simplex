# Core python libraries
import os
from typing import List, Callable, Optional
import uuid
# External libraries
import egttools as egt
from egttools.plotting.helpers import (xy_to_barycentric_coordinates,
                                       barycentric_to_xy_coordinates,
                                       find_roots_in_discrete_barycentric_coordinates,
                                       calculate_stability,
                                    )
from egttools.analytical.utils import (find_roots, check_replicator_stability_pairwise_games,)
from egttools.helpers.vectorized import vectorized_barycentric_to_xy_coordinates
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvals
from tqdm import tqdm

def run_all_simulations(param_list: list,
                        simulation_fn:callable=None,
                        plotting_fn:callable=None,
                        simulation_dir:str="",
                        plot_dir:str="",):
    """
    Iterate over each parameter dictionary, run the simulation, and save the results.

    Parameters:
    - param_list: A list of dictionaries, each containing a set of parameter values.
    """
    
    # Check if the output directory exists. If not, create it.
    if simulation_dir and not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    figs = []
    simulation_results = []
    # Construct a unique filename
    simulation_id = uuid.uuid4()
    for idx, parameters in tqdm(enumerate(param_list)):
        if simulation_fn is not None:
            df = simulation_fn(parameters)
            df["simulation_id"] = simulation_id
            df["model_id"] = idx
            # Save the dataframe to CSV
            filename = f"dataframe_{simulation_id}_{idx}.csv"
            filepath = os.path.join(simulation_dir, filename)
            df.to_csv(filepath, index=False)
            simulation_results.append(df)
        if plotting_fn is not None:
            fig = plotting_fn(parameters)
            # Save the figure
            filename = f"plot_{simulation_id}_{idx}.png"
            filepath = os.path.join(plot_dir, filename)
            fig.savefig(filepath)
            plt.close(fig)  # Close the figure to free up memory
            figs.append(fig)
        
    return figs, simulation_results

def replicator_equation(x: np.ndarray, payoff_function: callable, game_params: dict) -> np.ndarray:
    """
    Produces the discrete time derivative of the replicator dynamics with a dynamic payoff matrix.

    This only works for 2-player games.

    Parameters
    ----------
    x : numpy.ndarray[numpy.float64[m,1]]
        array containing the frequency of each strategy in the population.
    payoff_function : callable
        a function that calculates the payoff matrix given the current strategy frequencies and game parameters.
    game_params : dict
        dictionary containing the parameters of the game.

    Returns
    -------
    numpy.ndarray
        time derivative of x

    See Also
    --------
    egttools.analytical.StochDynamics
    egttools.numerical.PairwiseComparisonNumerical
    """
    
    # Calculate dynamic payoffs using the provided function
    payoffs = payoff_function(x, game_params)

    ax = np.dot(payoffs, x)
    x_dot = np.squeeze(x) * (np.squeeze(ax) - np.dot(np.squeeze(x), np.squeeze(ax)))

    return x_dot.reshape(-1, 1)

def vectorized_replicator_equation(frequencies: np.ndarray, payoff_function: callable, game_params: dict) -> np.ndarray:
    """
    Calculate gradients using barycentric coordinates in a strategy space.

    The input `frequencies` is a 3D tensor. The first dimension corresponds to the number of strategies (`p`), 
    and the subsequent `m x n` matrix corresponds to barycentric coordinates in the strategy space. Each row in this 
    matrix represents a specific set of barycentric coordinates, while the columns represent different scenarios 
    or replicates for that set of coordinates.

    Parameters
    ----------
    frequencies: numpy.ndarray[p, m, n]
        A 3D tensor where:
        - `p` is the number of strategies.
        - `m` corresponds to the number of rows of barycentric coordinates.
        - `n` corresponds to the number of columns or replicates of those coordinates.
        
    payoff_function : callable
        A function that calculates the payoff matrix given the current strategy frequencies and game parameters.
        
    game_params : dict
        Dictionary containing the parameters of the game.

    Returns
    -------
    numpy.ndarray[p, m, n]
        The gradients for each set of barycentric coordinates in the strategy space.
    """
    
    p, m, n = frequencies.shape
    gradients = np.zeros((p, m, n))
    for i in range(m):
        for j in range(n):
            current_freq = frequencies[:, i, j]
            # Calculate dynamic payoffs using the provided function
            payoffs = payoff_function(current_freq, game_params)
            
            ax = np.dot(payoffs, current_freq)
            
            # Calculating (ax - x * ax) for the current set of barycentric coordinates
            gradients[:, i, j] = current_freq * (ax - np.sum(current_freq * ax))
        
    return gradients

def check_replicator_stability_pairwise_games(stationary_points: List[np.ndarray],
                                              payoff_function: callable,
                                              game_params: dict,
                                              atol_neg: float = 1e-4,
                                              atol_pos: float = 1e-4,
                                              atol_zero: float = 1e-4) -> List[int]:
    """
    Calculates the stability of the roots assuming that they are from a system governed by the replicator
    equation (this function uses the Jacobian of the replicator equation in pairwise games to calculate the
    stability).

    Parameters
    ----------
    stationary_points: List[numpy.ndarray]
        a list of stationary points (represented as numpy.ndarray).
    payoff_matrix: numpy.ndarray
        a payoff matrix represented as a numpy.ndarray.
    atol_neg: float
        tolerance to consider a value negative.
    atol_pos: float
        tolerance to consider a value positive.
    atol_zero: float
        tolerance to determine if a value is zero.

    Returns
    -------
    List[int]
        A list of integers indicating the stability of the stationary points for the replicator equation:
        1 - stable
        -1 - unstable
        0 - saddle

    """

    def fitness(i: int, x: np.ndarray):
        payoff_matrix = payoff_function(x, game_params)
        return np.dot(payoff_matrix, x)[i]

    # First we build a Jacobian matrix
    def jacobian(x: np.ndarray):
        payoff_matrix = payoff_function(x, game_params)
        ax = np.dot(payoff_matrix, x)
        avg_fitness = np.dot(x, ax)
        jac = [[x[i] * (payoff_matrix[i, j]
                        - np.dot(x, payoff_matrix[:, j]))
                if i != j else (fitness(i, x)
                                - avg_fitness
                                + x[i] * (payoff_matrix[i, i]
                                          - np.dot(x, payoff_matrix[:, i])))
                for i in range(len(x))] for j in range(len(x))]
        return np.asarray(jac)

    stability = []

    for point in stationary_points:
        # now we check the stability of the roots using the jacobian
        eigenvalues = eigvals(jacobian(point))
        # Process eigenvalues to classify those within tolerance as zero
        effective_zero_eigenvalues = [ev for ev in eigenvalues if abs(ev.real) <= atol_zero]
        non_zero_eigenvalues = [ev for ev in eigenvalues if abs(ev.real) > atol_zero]

        print("point: ", point)
        print("eigenvalues: ", eigenvalues)
        print("effective_zero_eigenvalues: ", effective_zero_eigenvalues)
        print("non_zero_eigenvalues: ", non_zero_eigenvalues)
       # If all eigenvalues are effectively zero
        if len(effective_zero_eigenvalues) == len(eigenvalues):
            stability.append(0)  # Marginally or indeterminately stable
        # All non-zero eigenvalues have negative real parts => stable
        elif all(ev.real < -atol_neg for ev in non_zero_eigenvalues):
            stability.append(1)  # Stable
        # All non-zero eigenvalues have positive real parts => unstable
        elif all(ev.real > atol_pos for ev in non_zero_eigenvalues):
            stability.append(-1)  # Unstable
        # Mixture of positive and negative real parts => saddle
        else:
            stability.append(0)  # Saddle

    return stability

def tactical_deception_payoffs(freqs: np.ndarray, params: dict) -> np.ndarray:
    # For simplicity, just return a static matrix where the third strategy's payoff 
    # is inversely proportional to its frequency
    names = ["b", "c", "s", "d"]
    b, c, s, d = [params[k] for k in names]
    q = 1 - (freqs[2] / np.sum(freqs))
    payoffs = np.array([
        [b-c, -c*s, -c*(q + s - q*s)],
        [b*s, 0, 0],
        [b*(q + s - q*s) - d, -d, -d]
    ])
    return payoffs

def tactical_deception_payoffs_concave(freqs: np.ndarray, params: dict) -> np.ndarray:
    names = ["b", "c", "s", "d"]
    b, c, s, d = [params[k] for k in names]
    q = (1 - (freqs[2] / np.sum(freqs)))**0.1
    payoffs = np.array([
        [b-c, -c*s, -c*(q + s - q*s)],
        [b*s, 0, 0],
        [b*(q + s - q*s) - d, -d, -d]
    ])
    return payoffs

def tactical_deception_payoffs_convex(freqs: np.ndarray, params: dict) -> np.ndarray:
    names = ["b", "c", "s", "d"]
    b, c, s, d = [params[k] for k in names]
    q = 1 - (freqs[2] / np.sum(freqs))**10
    payoffs = np.array([
        [b-c, -c*s, -c*(q + s - q*s)],
        [b*s, 0, 0],
        [b*(q + s - q*s) - d, -d, -d]
    ])
    return payoffs

def tactical_deception_payoffs_s_curve(freqs: np.ndarray, params: dict) -> np.ndarray:
    names = ["b", "c", "s", "d"]
    b, c, s, d = [params[k] for k in names]
    bias = params.get("bias", 0.5)
    def tanh_variant(x, k=5):
        return (1 - np.tanh(k * (x - bias))) / 2
    q = tanh_variant(freqs[2] / np.sum(freqs))
    payoffs = np.array([
        [b-c, -c*s, -c*(q + s - q*s)],
        [b*s, 0, 0],
        [b*(q + s - q*s) - d, -d, -d]
    ])
    return payoffs

def pd_payoffs(freqs: np.ndarray, params: dict) -> np.ndarray:
    names = ["b", "c", "s", "d"]
    b, c, s, d = [params[k] for k in names]
    payoffs = np.array([
        [b-c, -c*s],
        [b*s, 0],
    ])
    return payoffs

def pd_payoffs_duped(freqs: np.ndarray, params: dict) -> np.ndarray:
    names = ["b", "c", "s", "d"]
    b, c, s, d = [params[k] for k in names]
    payoffs = np.array([
        [b-c, -c*s, -c*s],
        [b*s, 0, 0],
        [b*s, 0, 0]
    ])
    return payoffs

def plot_simplex(params):
    type_labels = params["strategies"]
    payoffs_fn = params["payoffs_fn"]
    simplex = egt.plotting.Simplex2D()
    frequencies = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    gradients = vectorized_replicator_equation(frequencies, payoffs_fn, params)
    xy_results = vectorized_barycentric_to_xy_coordinates(gradients, simplex.corners)
    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)
    
    # We also need to define some callable replicator equations for the plotting tools
    calculate_gradients = lambda u: replicator_equation(u, payoffs_fn, params)[:, 0]
    calculate_gradients_alt = lambda u, t: replicator_equation(u, payoffs_fn, params)[:, 0]

    
    # Compute equilibria and their stability
    roots = find_roots(gradient_function=calculate_gradients,
                    nb_strategies=frequencies.shape[0],
                    nb_initial_random_points=100)
    roots_xy = [barycentric_to_xy_coordinates(root, corners=simplex.corners) for root in roots]
    stability = check_replicator_stability_pairwise_games(roots, payoffs_fn, params)
    
    fig, ax = plt.subplots(figsize=(10,8))

    plot = (simplex.add_axis(ax=ax)
            .apply_simplex_boundaries_to_gradients(Ux, Uy)
            .draw_triangle()
            .draw_stationary_points(roots_xy, stability)
            .add_vertex_labels(type_labels)
            .draw_scatter_shadow(calculate_gradients_alt, 300, color='gray', marker='.', s=0.1)
            .draw_gradients()
            )

    ax.axis('off')
    ax.set_aspect('equal')

    plt.xlim((-.05,1.05))
    plt.ylim((-.02, simplex.top_corner + 0.05))
    return fig

def plot_simplex_numerical(params):
    
    names = ["Z", "beta", "mu", "strategies"]
    Z, beta, mu, strategies = [params[k] for k in names]
    type_labels = strategies
    simplex = egt.plotting.Simplex2D(discrete=True, size=Z, nb_points=Z+1)

    frequencies = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))

    frequencies_int = np.floor(frequencies * Z).astype(np.int64)

    # We make sure that our evolver represents payoffs as our desired function
    evolver_payoffs = lambda freqs: tactical_deception_payoffs(freqs, params)

    evolver = egt.analytical.StochDynamics(3, evolver_payoffs, Z)

    # We also need to ensure that any fitness calculations involving our payoffs
    # make use of our own custom methods.
    class_args = {"pop_size": Z,
                  "nb_strategies": len(type_labels),
                  "payoffs_fn": evolver_payoffs}
        
    def fitness_pair_functional(x: int, i: int, j: int, *args: Optional[list]) -> float:
            """
            Calculates the fitness of strategy i versus strategy j, in
            a population of x i-strategists and (pop_size-x) j strategists, considering
            a 2-player game.

            Parameters
            ----------
            x : int
                number of i-strategists in the population
            i : int
                index of strategy i
            j : int
                index of strategy j
            args : Optional[list]

            Returns
            -------
                float
                the fitness difference among the strategies
            """
            names = ["pop_size", "nb_strategies", "payoffs_fn"]
            pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
            popoulation_state_dict = {i: x, j: pop_size - x}
            population_state = [popoulation_state_dict.get(k, 0)
                                for k in range(nb_strategies)]
            payoff_matrix = payoffs_fn(population_state)
            fitness_i = ((x - 1) * payoff_matrix[i, i] +
                        (pop_size - x) * payoff_matrix[i, j]) / (pop_size - 1)
            fitness_j = ((pop_size - x - 1) * payoff_matrix[j, j] +
                        x * payoff_matrix[j, i]) / (pop_size - 1)
            return fitness_i - fitness_j

    def full_fitness_difference_pairwise_functional(i: int, j: int, population_state: np.ndarray) -> float:
            """
            Calculates the fitness of strategy i in a population with state :param population_state,
            assuming pairwise interactions (2-player game).

            Parameters
            ----------
            i : int
                index of the strategy that will reproduce
            j : int
                index of the strategy that will die
            population_state : numpy.ndarray[numpy.int64[m,1]]
                            vector containing the counts of each strategy in the population

            Returns
            -------
            float
            The fitness difference between the two strategies for the given population state
            """
            names = ["pop_size", "nb_strategies", "payoffs_fn"]
            pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
            # Here, our payoffs depend on the population state.
            payoff_matrix = payoffs_fn(population_state)
            fitness_i = (population_state[i] - 1) * payoff_matrix[i, i]
            for strategy in range(nb_strategies):
                if strategy == i:
                    continue
                fitness_i += population_state[strategy] * payoff_matrix[i, strategy]
            fitness_j = (population_state[j] - 1) * payoff_matrix[j, j]
            for strategy in range(nb_strategies):
                if strategy == j:
                    continue
                fitness_j += population_state[strategy] * payoff_matrix[j, strategy]

            return (fitness_i - fitness_j) / (pop_size - 1)

    evolver.full_fitness = full_fitness_difference_pairwise_functional
    evolver.fitness = fitness_pair_functional

    result = np.asarray([[evolver.full_gradient_selection(frequencies_int[:, i, j], beta)
                        for j in range(frequencies_int.shape[2])]
                        for i in range(frequencies_int.shape[1])]).swapaxes(0, 1).swapaxes(0, 2)

    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)
    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)

    calculate_gradients = lambda u: Z*evolver.full_gradient_selection(u, beta)

    roots = find_roots_in_discrete_barycentric_coordinates(calculate_gradients, Z, nb_interior_points=5151, atol=1e-1)
    roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]

    stability = calculate_stability(roots, calculate_gradients)

    evolver.mu = 0
    sd_rare_mutations = evolver.calculate_stationary_distribution(beta)
    print("Stationary Distribution of the 3 strategies: ", sd_rare_mutations)

    evolver.mu = mu
    sd = evolver.calculate_stationary_distribution(beta)

    fig, ax = plt.subplots(figsize=(15,10))

    plot = (simplex.add_axis(ax=ax)
            .apply_simplex_boundaries_to_gradients(Ux, Uy)
            .draw_gradients(zorder=5)
            .add_colorbar()
            .draw_stationary_points(roots_xy, stability, zorder=11)
            .add_vertex_labels(type_labels)
            .draw_stationary_distribution(sd, vmax=0.0001, alpha=0.5, edgecolors='gray', cmap='binary', shading='gouraud', zorder=0)
            )

    ax.axis('off')
    ax.set_aspect('equal')

    plt.xlim((-.05,1.05))
    plt.ylim((-.02, simplex.top_corner + 0.05))

    return fig

def plot_deception_frequency(params):
    
    names = ["Z", "beta", "mu", "strategies"]
    Z, beta, mu, strategies = [params[k] for k in names]
    type_labels = strategies

    # We make sure that our evolver represents payoffs as our desired function
    evolver_payoffs = lambda freqs: tactical_deception_payoffs(freqs, params)

    evolver = egt.analytical.StochDynamics(3, evolver_payoffs, Z)

    # We also need to ensure that any fitness calculations involving our payoffs
    # make use of our own custom methods.
    class_args = {"pop_size": Z,
                  "nb_strategies": len(type_labels),
                  "payoffs_fn": evolver_payoffs}
        
    def fitness_pair_functional(x: int, i: int, j: int, *args: Optional[list]) -> float:
            """
            Calculates the fitness of strategy i versus strategy j, in
            a population of x i-strategists and (pop_size-x) j strategists, considering
            a 2-player game.

            Parameters
            ----------
            x : int
                number of i-strategists in the population
            i : int
                index of strategy i
            j : int
                index of strategy j
            args : Optional[list]

            Returns
            -------
                float
                the fitness difference among the strategies
            """
            names = ["pop_size", "nb_strategies", "payoffs_fn"]
            pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
            popoulation_state_dict = {i: x, j: pop_size - x}
            population_state = [popoulation_state_dict.get(k, 0)
                                for k in range(nb_strategies)]
            payoff_matrix = payoffs_fn(population_state)
            fitness_i = ((x - 1) * payoff_matrix[i, i] +
                        (pop_size - x) * payoff_matrix[i, j]) / (pop_size - 1)
            fitness_j = ((pop_size - x - 1) * payoff_matrix[j, j] +
                        x * payoff_matrix[j, i]) / (pop_size - 1)
            return fitness_i - fitness_j

    def full_fitness_difference_pairwise_functional(i: int, j: int, population_state: np.ndarray) -> float:
            """
            Calculates the fitness of strategy i in a population with state :param population_state,
            assuming pairwise interactions (2-player game).

            Parameters
            ----------
            i : int
                index of the strategy that will reproduce
            j : int
                index of the strategy that will die
            population_state : numpy.ndarray[numpy.int64[m,1]]
                            vector containing the counts of each strategy in the population

            Returns
            -------
            float
            The fitness difference between the two strategies for the given population state
            """
            names = ["pop_size", "nb_strategies", "payoffs_fn"]
            pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
            # Here, our payoffs depend on the population state.
            payoff_matrix = payoffs_fn(population_state)
            fitness_i = (population_state[i] - 1) * payoff_matrix[i, i]
            for strategy in range(nb_strategies):
                if strategy == i:
                    continue
                fitness_i += population_state[strategy] * payoff_matrix[i, strategy]
            fitness_j = (population_state[j] - 1) * payoff_matrix[j, j]
            for strategy in range(nb_strategies):
                if strategy == j:
                    continue
                fitness_j += population_state[strategy] * payoff_matrix[j, strategy]

            return (fitness_i - fitness_j) / (pop_size - 1)

    evolver.full_fitness = full_fitness_difference_pairwise_functional
    evolver.fitness = fitness_pair_functional

    # Enforce mu=0, otherwise, we would be computing the full stationary
    # distribution of the population state for every value of beta.
    # We just want the stationary distribution with respect to the monomorphic
    # strategy profiles.
    evolver.mu = 0
    beta_list = np.arange(0.01, 10, 0.01)
    stationary_distributions = []
    for _, beta in tqdm(enumerate(beta_list)):
        sd = evolver.calculate_stationary_distribution(beta)
        stationary_distributions.append(sd)
        
    deception_frequency = [sd[-1] for sd in stationary_distributions]

    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(beta_list,
             deception_frequency,
             label='Deception Frequency',
             color='blue',
             linewidth=2)
    plt.title('Deception Frequency vs Selection Intensity')
    plt.xlabel('Selection Intensity (beta)')
    plt.ylabel('Deception Frequency (x_td)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return fig

def plot_strategy_frequencies(params):
    
    names = ["Z", "beta", "mu", "strategies"]
    Z, beta, mu, strategies = [params[k] for k in names]
    type_labels = strategies

    # We make sure that our evolver represents payoffs as our desired function
    evolver_payoffs = lambda freqs: tactical_deception_payoffs(freqs, params)

    evolver = egt.analytical.StochDynamics(3, evolver_payoffs, Z)

    # We also need to ensure that any fitness calculations involving our payoffs
    # make use of our own custom methods.
    class_args = {"pop_size": Z,
                  "nb_strategies": len(type_labels),
                  "payoffs_fn": evolver_payoffs}
        
    def fitness_pair_functional(x: int, i: int, j: int, *args: Optional[list]) -> float:
            """
            Calculates the fitness of strategy i versus strategy j, in
            a population of x i-strategists and (pop_size-x) j strategists, considering
            a 2-player game.

            Parameters
            ----------
            x : int
                number of i-strategists in the population
            i : int
                index of strategy i
            j : int
                index of strategy j
            args : Optional[list]

            Returns
            -------
                float
                the fitness difference among the strategies
            """
            names = ["pop_size", "nb_strategies", "payoffs_fn"]
            pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
            popoulation_state_dict = {i: x, j: pop_size - x}
            population_state = [popoulation_state_dict.get(k, 0)
                                for k in range(nb_strategies)]
            payoff_matrix = payoffs_fn(population_state)
            fitness_i = ((x - 1) * payoff_matrix[i, i] +
                        (pop_size - x) * payoff_matrix[i, j]) / (pop_size - 1)
            fitness_j = ((pop_size - x - 1) * payoff_matrix[j, j] +
                        x * payoff_matrix[j, i]) / (pop_size - 1)
            return fitness_i - fitness_j

    def full_fitness_difference_pairwise_functional(i: int, j: int, population_state: np.ndarray) -> float:
            """
            Calculates the fitness of strategy i in a population with state :param population_state,
            assuming pairwise interactions (2-player game).

            Parameters
            ----------
            i : int
                index of the strategy that will reproduce
            j : int
                index of the strategy that will die
            population_state : numpy.ndarray[numpy.int64[m,1]]
                            vector containing the counts of each strategy in the population

            Returns
            -------
            float
            The fitness difference between the two strategies for the given population state
            """
            names = ["pop_size", "nb_strategies", "payoffs_fn"]
            pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
            # Here, our payoffs depend on the population state.
            payoff_matrix = payoffs_fn(population_state)
            fitness_i = (population_state[i] - 1) * payoff_matrix[i, i]
            for strategy in range(nb_strategies):
                if strategy == i:
                    continue
                fitness_i += population_state[strategy] * payoff_matrix[i, strategy]
            fitness_j = (population_state[j] - 1) * payoff_matrix[j, j]
            for strategy in range(nb_strategies):
                if strategy == j:
                    continue
                fitness_j += population_state[strategy] * payoff_matrix[j, strategy]

            return (fitness_i - fitness_j) / (pop_size - 1)

    evolver.full_fitness = full_fitness_difference_pairwise_functional
    evolver.fitness = fitness_pair_functional

    # Enforce mu=0, otherwise, we would be computing the full stationary
    # distribution of the population state for every value of beta.
    # We just want the stationary distribution with respect to the monomorphic
    # strategy profiles.
    evolver.mu = 0
    beta_list = np.arange(0.01, 10, 0.01)
    stationary_distributions = []
    for _, beta in tqdm(enumerate(beta_list)):
        sd = evolver.calculate_stationary_distribution(beta)
        stationary_distributions.append(sd)
    strategy_1_freq = [sd[0] for sd in stationary_distributions]
    strategy_2_freq = [sd[1] for sd in stationary_distributions]
    strategy_3_freq = [sd[2] for sd in stationary_distributions]
    
    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    # Plotting each strategy's frequency
    plt.plot(beta_list, strategy_1_freq, label='CC', color='blue', linewidth=2)
    plt.plot(beta_list, strategy_2_freq, label='HD', color='red', linewidth=2)
    plt.plot(beta_list, strategy_3_freq, label='TD', color='green', linewidth=2)
    
    # Setting the title and labels
    plt.title('Strategy Frequencies vs Selection Intensity')
    plt.xlabel('Selection Intensity (beta)')
    plt.ylabel('Strategy Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return fig
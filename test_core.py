# Core python libraries
import unittest
# External libraries
import numpy as np
# Project imports
import core

# Example payoff function
def example_payoff_function(x: np.ndarray, game_params: dict) -> np.ndarray:
    # This is a placeholder function. For this example, I'm just returning a transposed matrix.
    return np.array([[1, 3], [2, 4]])

class TestReplicatorEquation(unittest.TestCase):

    def test_replicator_equation(self):
        x = np.array([[0.5], [0.5]])
        game_params = {}  # Add game parameters here as needed
        
        result = core.replicator_equation(x, example_payoff_function, game_params)
        expected = np.array([[-0.25], [0.25]])

        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_vectorized_replicator_equation(self):
        # Define a simple dynamic payoff function
        def sample_payoff_function(freqs: np.ndarray, params: dict) -> np.ndarray:
            # For simplicity, just return a static matrix where the third strategy's payoff 
            # is inversely proportional to its frequency
            return np.array([
                [1, 2, 1 - freqs[2]],
                [2, 3, 2 * (1 - freqs[2])],
                [1, 1, 3]
            ])

        # For this example, let's assume:
        # 3 strategies (for a 2D simplex)
        # 4 elements for each barycentric coordinate
        # 2 barycentric coordinates
        frequencies = np.random.rand(3, 3, 2)  

        # Call the function
        gradients = core.vectorized_replicator_equation(frequencies, sample_payoff_function, {})

        # Validate the shape of the result
        assert gradients.shape == (3, 3, 2), f"Expected shape (3, 4, 2), but got {gradients.shape}"
        
        def tactical_deception_payoffs(freqs: np.ndarray, params: dict) -> np.ndarray:
            # For simplicity, just return a static matrix where the third strategy's payoff 
            # is inversely proportional to its frequency
            names = ["b", "c", "s", "d"]
            b, c, s, d = [params[k] for k in names]
            q = 1 - (freqs[2] / np.sum(freqs))
            return np.array([
                [b-c, -c*s, -c*(q + s - q*s)],
                [b*s, 0, 0],
                [b*(q + s - q*s) - d, -d, -d]
            ])

        # For this example, let's assume:
        # 3 strategies (for a 2D simplex)
        # 4 elements for each barycentric coordinate
        # 2 barycentric coordinates
        frequencies = np.random.rand(3, 3, 2)  
        params = {"b": 2, "c": 0.5, "d": 0.2, "s": 0.1}
        # Call the function
        gradients = core.vectorized_replicator_equation(frequencies, tactical_deception_payoffs, params)

        # Validate the shape of the result
        assert gradients.shape == (3, 3, 2), f"Expected shape (3, 3, 2), but got {gradients.shape}"

if __name__ == "__main__":
    unittest.main()
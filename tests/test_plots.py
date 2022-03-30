import os
import unittest
from unittest import TestCase

import jax.numpy as jnp
from numpy import loadtxt

from ghgf.model import HGF

path = os.path.dirname(os.path.abspath(__file__))


class Testsdt(TestCase):
    def test_plot_trajectories(self):

        # Set up standard 3-level HGF for continuous inputs
        hgf = HGF(
            n_levels=3,
            model_type="GRW",
            initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_pi={"1": 1e4, "2": 1e1, "3": 1e1},
            omega={"1": -13.0, "2": -2.0, "3": -2.0},
            rho={"1": 0.0, "2": 0.0, "3": 0.0},
            kappa={"1": 1.0, "2": 1.0},
        )

        # Read USD-CHF data
        timeserie = loadtxt("/home/nicolas/git/ghgf/tests/data/usdchf.dat")
        data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T

        # Feed input
        hgf.input_data(input_data=data)

        # Plot
        for backend in ["matplotlib", "bokeh"]:
            hgf.plot_trajectories(backend=backend)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

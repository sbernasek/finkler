import numpy as np
from functools import reduce
from operator import add

# load timeseries container
from .timeseries import PerturbationTimeSeries
from .dimensioned import Network

# load solver components
from solver.simulate import MonteCarloSimulation
from solver.solver.signals import cSquarePulse


class Simulation:
    """
    Class defines a dimensioned simulation.

    Attributes:
    solver (solver.MonteCarloSimulation) - dimensioned simulation SSA solver
    ic (1-D array of ints) - initial condition
    """

    def __init__(self, network, ic=None):
        """
        Args:
        network (Network instance) - solver-compatible network
        ic (1-D array) - initial conditions for solver (defaults to zeros)
        """
        self.solver = MonteCarloSimulation(network, ic=ic)

    @staticmethod
    def from_gnw(gnw_simulation, T=1, X=100, Y=100):
        """
        Build from GNW simulation.

        Args:
        gnw_simulation (gnw.Simulation)
        """

        # build dimensioned network and instantiate simulator
        network = Network.from_sbml(gnw_simulation.sbml, T=T, X=X, Y=Y)

        # dimensionalize steady states
        ssr = X * gnw_simulation.ssrx
        ssp = Y * gnw_simulation.sspx
        ss = np.hstack((ssr, ssp)) #/ gnw_sim.norm

        # rearrange states to simulator order
        ind = np.array(reduce(add, zip(range(4), range(4, 8))))
        ic = ss[ind]

        return Simulation(network, ic=ic)

    def run(self, ptb, num_trials=3, duration=1000, dt=1):
        """
        Run simulation for single perturbation.

        Args:
        ptb (float) - perturbation value
        num_trials (int) - number of stochastic trajectories
        duration (float) - simulation duration
        dt (float) - sampling interval

        Returns:
        ts (TimeSeries instance) - simulation output
        """

        # define perturbation signal
        perturbation = cSquarePulse(t_on=0, t_off=duration/2, off=0, on=ptb)

        # run simulation
        ts = self.solver.run(input_function=perturbation,
                     num_trials=num_trials,
                     duration=duration,
                     dt=dt)

        return ts

    def run_multiple(self, perturbations, **kw):
        """
        Run simulation for multiple perturbations.

        Args:
        perturbations (array like) - perturbation values
        kw - keywork arguments for Simulation.run

        Returns
        ts (PerturbationTimeSeries instance) - simulation output
        """
        ts = [self.run(ptb, **kw) for ptb in perturbations]
        return PerturbationTimeSeries.from_timeseries_list(ts)

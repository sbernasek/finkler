import matplotlib.pyplot as plt


def PlotAllTimeSeries(ts):
    """
    Plot all species for a given simulation.

    Args:
    ts (PerturbationTimeSeries) - simulation output
    """

    # plot both timeseries results
    fig, axes = plt.subplots(nrows=8, ncols=1, sharey=False, figsize=(3, 15))

    for species, ax0 in enumerate(axes):
        ts.plot_perturbations(species, ax=ax0)
        #ax0.set_ylim(0, 1.5)
        ax0.set_xticks([])
        ax0.set_xlabel('')

    # format axes
    axes[0].set_title('Simulation')

    plt.tight_layout()

    return fig


def ValidateSpecies(reference, reproduction, species=0):
    """
    Compare single species between GNW reference and dimensioned reproduction.

    Args:
    reference (PerturbationTimeSeries) - GNW simulation output
    reproduction (PerturbationTimeSeries) - SSA solver output
    species (int) - species index
    """

    # plot both timeseries results
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(6, 2))
    ax0, ax1 = axes
    reference.plot_perturbations(species, ax=ax0)
    reproduction.plot_perturbations(species, ax=ax1)

    # format axes
    ax0.set_title('GNW Simulation')
    ax1.set_title('Dimensioned simulation')
    ax0.set_ylim(0, 1.5)

    plt.tight_layout()

    return fig


def ValidateAll(reference, reproduction):
    """
    Compare all species between GNW reference and dimensioned reproduction.

    Args:
    reference (PerturbationTimeSeries) - GNW simulation output
    reproduction (PerturbationTimeSeries) - SSA solver output
    """

    # plot both timeseries results
    fig, axes = plt.subplots(nrows=8, ncols=2, sharey=True, figsize=(6, 15))

    for species, axrow in enumerate(axes):
        ax0, ax1 = axrow
        reference.plot_perturbations(species, ax=ax0)
        reproduction.plot_perturbations(species, ax=ax1)
        ax0.set_ylim(0, 1.5)
        ax1.set_ylabel('')
        ax0.set_xticks([])
        ax1.set_xticks([])
        ax0.set_xlabel('')
        ax1.set_xlabel('')

    # format axes
    ax0, ax1 = axes[0]
    ax0.set_title('GNW Simulation')
    ax1.set_title('Dimensioned simulation')

    plt.tight_layout()

    return fig


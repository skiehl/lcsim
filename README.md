# lcsim
A python3 package to create artificial blazar light curves with a predefined
power spectral density (PSD) and probability density function (PDF).

## Requirements

The `lcsim` script uses the following standard python packages:

+ copy
+ math

The `lcsim` script uses the following python packages:

+ numpy
+ scipy
+ statsmodels

The `simdb` script uses the following standard python packages:

+ os
+ sys

The `simdb` script uses the following python packages:

+ numpy
+ sqlite3

## Getting Started

Get the python scripts:

    $ git clone https://github.com/skiehl/lcsim.git

## Modules

+ `lcsim.py`: The main module for simulating light curves.
+ `simdb.py`: A SQLite3-based database backend to store simulations.
+ `run_lcsim.py`: A wrapper script to run and save light curve simulations for
  multiple sources.

## Usage

Usage of the package is demonstrated in the jupyter notebook
`SimulationDemo.ipynb`

*Note:* Documentation of the simdb.py module is not yet included.

## References

[Timmer&Koenig, 1995](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..707T/abstract).
[Emmanoulopoulos et al., 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.433..907E/abstract).

## License

lcsim is licensed under the BSD 3-Clause License - see the
[LICENSE](https://github.com/skiehl/lcsim/blob/master/LICENSE) file.

## Alternatives

At least on other python implementations of the Emmanoulopoulos+ method is available:

+ [ DELightcurveSimulation](https://github.com/samconnolly/DELightcurveSimulation)

<a href="https://ascl.net/2310.002"><img src="https://img.shields.io/badge/ascl-2310.002-blue.svg?colorB=262255" alt="ascl:2310.002" /></a>

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

Open the jupyter notebook `SimulationDemo.ipynb` for a demonstation of the
code. Either use jupyter lab:

    $ jupyter lab

Or jupyter notebook:

    $ jupyter notebook SimulationDemo.ipynb

## Modules

+ `lcsim.py`: The main module for simulating light curves.
+ `simdb.py`: A SQLite3-based database backend to store simulations.
+ `run_lcsim.py`: A wrapper script to run and save light curve simulations for
  multiple sources.

## Usage

Usage of the package is demonstrated in the jupyter notebook
`SimulationDemo.ipynb`

*Note:* Documentation of the simdb.py module is not yet included.

## Citation

Bibtex:

```
@MISC{2023ascl.soft10002K,
       author = {{Kiehlmann}, Sebastian},
        title = "{lcsim: Light curve simulation code}",
     keywords = {Software},
 howpublished = {Astrophysics Source Code Library, record ascl:2310.002},
         year = 2023,
        month = oct,
          eid = {ascl:2310.002},
        pages = {ascl:2310.002},
archivePrefix = {ascl},
       eprint = {2310.002},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023ascl.soft10002K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


This software package is indexed on [ASCL](https://ascl.net/2310.002) and [ADS](https://ui.adsabs.harvard.edu/abs/2023ascl.soft10002K/).

## License

lcsim is licensed under the BSD 3-Clause License - see the
[LICENSE](https://github.com/skiehl/lcsim/blob/master/LICENSE) file.

## References

[Timmer&Koenig, 1995](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..707T/abstract).
[Emmanoulopoulos et al., 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.433..907E/abstract).

## Alternatives

At least on other python implementations of the Emmanoulopoulos+ method is available:

+ [ DELightcurveSimulation](https://github.com/samconnolly/DELightcurveSimulation)

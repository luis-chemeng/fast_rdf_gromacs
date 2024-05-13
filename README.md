# Fast RDF computation from GROMACS simulation files
A small Python module that simplifies the calculation of radial distribution functions from GROMACS simulation results.

## Example

Let's create and activate a conda environment:

```
conda create -n RDF
```

```
conda activate RDF
```

The computations of RDF are made by the "mdtraj" module (mdtraj.org), so install this dependency:

```
conda install -c conda-forge mdtraj
```

Let's install "matplotlib" to graph the results:

```
 conda install conda-forge::matplotlib
```

Take a look of the example.py file and run it:

```
 pyhton example.py
```

This generates RDF graph representing some interesting interactions within a mixture of water and ethanol.

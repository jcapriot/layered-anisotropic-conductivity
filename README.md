Anisotropic clustering inversion
================================

This repository contains python and jupyter notebooks demonstrating how to use
clustering inversion to solve the inverse problem of a fully anisotropic 
conductivity layered earth, measured by the electrical soundings. The purpose of 
this work is to demonstrate a repeatable method of applying clustering inversion 
to an anisotropic quantity when incorporating petrophysical information.

Requirements
------------
To run these codes the only requirements are:

* `numpy`
* `scipy`
* `jupyter`

Anisotropic layered half-space
------------------------------
The code implementation of the electrical potential due to a layered half-space
is found in the ``anisotropic_potential.py``. In it are the necessary steps to
model the voltage for given combindations of A, B, M, and N electrodes, using a
semi-analytic form. This formulation is analytic in the wavenumber domain, at
which point it uses a real ifft. The jacobian of the kernel is calculated using
a reverse mode operation that has a cost of roughly 2x the forward operation.


Inversion Notebooks
-------------------
There are four notebooks that correspond to the examples presented in the
formal report of this method. The ``Inversion`_cross.ipynb`` corresponds to the 
main example used throughout the work. The three other notebooks, 
``Inversion1_star_cross.ipynb``, ``Inversion2.ipynb``, and ``Inversion3.ipynb`` 
then correspond to the three repeated examples which show the generality of the 
approach to different amount of data and with different true models.

Plotting Notebook
-----------------
The ``Plotting.ipynb`` notebook contains all of the figures prepared for the 
manuscript.

Models
------
All output models and data for each of the four different cases can be found in the `Models` directory. These will also be recreated by running the Inversion notebooks.
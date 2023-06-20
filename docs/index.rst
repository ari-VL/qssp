.. qssp documentation master file, created by
   sphinx-quickstart on Wed Apr 12 15:08:47 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. py:module:: qssp

************************************************
'qssp': quantum-state stochastic processes
************************************************

Introduction
------------

Quantum-state stochastic processes (qssp's) are
quantum objects which inherit a probabilistic
structure from some underlying classical process.
Using this package one can generate a qssp by 
combining a finite-state hidden Markov model (HMM)
with an alphabet of pure quantum states. One can
then (1) calculate their quantum information
properties directly or (2) apply a set of POVMs
to obtain a classical measured process and then
calculate the classical information properties
of the measured process.

This code was used for calculations in the
following papers:

Contents:

.. toctree::
   :maxdepth: 2

   generalinfo
   modules
   examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

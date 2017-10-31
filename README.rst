:shrimp:: Finding small bodied problems that are everywhere
===========================================================

Package to visualize common trends in Kepler/K2 data including rolling band noise and common periodic signals across channels.

Usage
=====

.. highlight:: python
    from krill import krill
    k = krill(campaign=6)
    k.build_cad()
    k.build(channel=44)
    k.power()     #Build Power Spectrum
    k.rolling()   #Build rolling band image

    
Dependencies
------------
k2movie

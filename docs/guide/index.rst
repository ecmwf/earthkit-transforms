User guide
==========

**earthkit-transforms** is a library of software tools to support people
working with climate and meteorology data.It is made of sub-modules desigined
for  transformations of data in specific domains.
For example, the temporal module contains methods for aggregation and
statistics analysis accross time dimensions/coordinates.

earthkit.transforms makes computations in given domains easier by detecting the dimensions and coordinates
of the input data in which it operates. The detection is based on CF conventions, and some additional
common identifiers. Users are encouraged to ensure their data is CF compliant to get the best experience.
For more details see the :doc:`dimension detection guide <dimension-detection>`.

.. toctree::
   :maxdepth: 1

   spatial
   temporal
   climatology
   ensemble
   dimension-detection

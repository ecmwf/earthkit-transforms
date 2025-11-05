Version 0.5 Updates
/////////////////////////

Version 0.5.0
===============

**Major change (#61)**: The module API has been updated such that the intermediate module aggregate has been deprecated.
earthkit.transforms now contains submodules which are relevant to the dimensions/coordinates of computation,
e.g. earthkit.transforms.temporal and earthkit.transforms.spatial. All methods associated with these
dimensions/coordinates will be found within, at present this is only the existing aggregation methods,
but will now contain other things such as trend analysis etc. Also note that the old API (earthkit.transforms.aggregate.XXX) will continue to work, but will produce
deprecation warnings, it will be fully removed in version 2.0 of earthkit.transforms.
However, you are advised to remove sooner as tests will not necessary cover the deprecated API.

Other updates include:

- Array api updates for gpu by @EddyCMWF in #51
- simplify spatial.py by @Oisin-M in #55
- Add ensemble aggregation methods by @JamesVarndell in #52
- Release action update by @EddyCMWF in #64, #66, #67



Version 0.5.1 - 0.5.3
=====================

- tools submodule revert by @EddyCMWF in #70
- Hotfix backwards compatibility, ensure that earthkit.transforms.aggregate is loaded with import of earthkit.transforms. Deprecation warnings applied more targetedly by @EddyCMWF in #71

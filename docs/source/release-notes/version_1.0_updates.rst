Version 1.0 Updates
/////////////////////////


Version 1.0.0
=============

**Major release**: This release marks the point at which we consider
the API to be stable and ready for production use. We will continue to make improvements and add features,
but we will strive to maintain backward compatibility as much as possible. We will also be updating our
documentation and examples to reflect the new features and improvements in this release.

**Breaking changes**: The default arguments for the functions in the `earthkit.transforms.climatology` module
have been updated. Please refer to the documentation and examples for the usage.


Other changes:

* Add to downstream ci 3 by @EddyCMWF in #76, #77, #78
* Address index change in pandas by @EddyCMWF in #79
* Add acccumulation-to-rate and deaccumulate functions to temporal sub-package by @EddyCMWF in #75
* `extra_reduce_dims` in temporal sub-package by @EddyCMWF in #84 and #85
* Update to linting tools by @mcocdawc in #87
* Standardised interfaces and decorators for version1.0 release by @EddyCMWF in #88
* Remove direct ek-data dependency by @EddyCMWF in #89
* Update anomaly default behaviour by @EddyCMWF in #90
* Update docs by @EddyCMWF in #95, #97, #98, #99, #100, #101, #102

New Contributors
----------------

* @mcocdawc made their first contribution in #87

**Full Changelog**: https://github.com/ecmwf/earthkit-transforms/compare/0.5.4...1.0.0

**Github release-notes**: https://github.com/ecmwf/earthkit-transforms/releases/tag/1.0.0
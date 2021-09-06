
======================
What's new in ESEm 1.1
======================

This page documents the new features added, and bugs fixed in ESEm since version 1.0. For more detail see all changes here: https://github.com/duncanwp/ESEm/compare/1.0.0...1.1.0

ESEm 1.1 features
=================

 * We have added this What's New page for tracking the latest developments in ESEm!
 * We have dropped the mandatory requirement of Iris to make installation of ESEm easier. We have also added support for
   xarray DataArrays so that users can use their preferred library for data processing.
 * The :meth:`esem.emulator.Emulator.predict` and :meth:`esem.emulator.Emulator.batch_stats` methods can now accept pd.DataFrames to match the training interface.
   The associated doc-strings and signatures have been extended to reflect this.

Bugs fixed
==========

 * Use tqdm.auto to automatically choose the appropriate progress bar for the context
 * Fix `plot_validation` handling of masked data


===========
ESEm design
===========

Here we provide a brief description of the main architectural decisions behind the design for ESEm in order
to hopefully make it easier for contributors and users alike to understand the various components and how they
fit together.

Emulation
=========

We try to provide a seamless interface for users whether they provide iris Cube's or numpy ndarrays. This is
done using the :class:`esem.cube_wrapper.CubeWrapper` which keeps a copy of the provided cube but only exposes the
underlying numpy array to the emulation engines. When the data is requested from this wrapper using the
:meth:`esem.cube_wrapper.CubeWrapper.wrap` method then it will return a copy of the input cube with the data
replaced by the emulated data, or if no cube was provided then just the data.

This layer will also ensure the (numpy) data is wrapped in a :class:`esem.cube_wrapper.DataWrapper`. This class
transparently applies any requested :class:`esem.data_processors.DataProcessor` in sequence.

The user can then create an :class:`esem.emulator.Emulator` object by providing a concrete
:class:`esem.model_adaptor.ModelAdaptor` such as a :class:`esem.model_adaptor.KerasModel`. There are two layers of
abstraction here: The first to deal with different interfaces to different emulation libraries; and the second to apply
the pre- and post-processing and allow a single :meth:`esem.emulator.Emulator.batch_stats` method. The
:meth:`esem.emulator.Emulator._predict` provides an important internal interface to the underlying model which reverts
any data-processing but leaves the emulator output as a TensorFlow Tensor to allow optimal sampling.

The top-level functions :func:`esem.gp_model`, :func:`esem.cnn_model` and :func:`esem.rf_model` provide an simple 
interface for constructing these emulators and should be sufficient for most users.

Calibration
===========

We try and keep this interface very simple; a :class:`esem.sampler.Sampler` should be initialised with an
:class:`esem.emulator.Emulator` object to sample from, some observations and associated uncertainties. The only method
it has to provide is :meth:`esem.sampler.Sampler.sample` which should provide sample :math:`\theta` from the posterior.

Wherever possible these samplers should take advantage of the fact that the :meth:`esem.emulator.Emulator._predict`
method returns TensorFlow tensors and always prefer to use them directly rather than using :meth:`esem.emulator.Emulator.predict`
or calling `.numpy()` on them. This allows the sampling to happen on GPUs where available and can substantially speed-up sampling.

The :class:`esem.abc_sampler.ABCSampler` extends this interface to include both
:meth:`esem.abc_sampler.ABCSampler.get_implausibility` and :meth:`esem.abc_sampler.ABCSampler.batch_constrain` methods.
The first allows inspection of the effect of different observations on the constraint and the second allows a streamlined
approach for rejecting samples in batch, taking advantage of the large amounts of memory available on modern GPUs.

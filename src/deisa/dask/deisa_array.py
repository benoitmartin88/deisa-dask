# =============================================================================
# Copyright (C) 2026 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
The public array type a DEISA callback receives: one ``DeisaArray`` per registered array per timestep.

The base contract of the public API is that a callback takes a LIST OF ``DeisaArray``: ``def cb(window)`` receives a
sliding window whose entries are the array delivered at each timestep, so ``window[-1]`` is the current timestep and
``window[-1].sum()`` a reduction of it. Both delivery modes hand over the same class, which is what makes the callback
mode-agnostic:

- **legacy full-chunk scatter** (``precompute=False``): a plain :class:`DeisaArray` -- a ``dask.array.Array``
  subclass carrying ``.timestep`` -- over the real graph of gathered chunk futures, exactly as before the reduction
  path existed.
- **on-demand reduction delivery** (``precompute=True``, the default): an :class:`OndemandDeisaArray`. It IS a
  :class:`DeisaArray` (same name, same ``isinstance``), but it carries no data: its graph is empty by construction
  and refuses to be read, and its operator surface is the closed lazy one of :mod:`deisa.dask.ondemand_expr`, whose
  ``.compute()`` dispatches a plan to the bridges instead of gathering the field.

Why the two cannot be one class body: ``dask.array.Array`` semantics and lazy-operand semantics CONFLICT for the same
object. A legacy array must support real dask operations (its graph is over chunk futures) and numpy interop; a
data-free array must refuse both, because materializing it is exactly the full-chunk gather the reduction path exists
to avoid. The subclass swap below keeps both: :class:`OndemandDeisaArray` puts :class:`~deisa.dask.ondemand_expr.
LazyOperand` FIRST in its MRO, so the guards and the closed surface win over dask's, and it refuses, one by one, the
dask ``Array`` methods that would otherwise build a graph over a field that was never delivered.

The refusal wording is deliberately UNCHANGED from the pre-K13C carrier (it says "the collection ..."): the messages
are part of the refusal contract asserted by the test suite, and no refusal semantics may move in this change.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

import dask
import dask.array as da
from deisa.dask import ondemand_errors as errors
from deisa.dask.ondemand_expr import _CHUNK_PROGRAM, LazyOperand, Root

__all__ = ["DeisaArray", "OndemandDeisaArray"]


# =============================================================================
# DeisaArray -- copied from deisa-core
#
# TODO(ben): deisa-core will import this class from deisa-dask instead of defining it; drop this copy then. The body
# below is ``deisa.core.types.DeisaArray`` verbatim (same ``t``/``timestep``, same copy/deepcopy, same HDF5 helpers),
# so an operand built here is interchangeable with the deisa-core one as soon as that swap lands.
# =============================================================================
class DeisaArray(da.Array):
    """A dask array carrying the simulation timestep it was delivered for."""

    def __new__(cls, t, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, t, *args, **kwargs):
        self.t = t

    def copy(self):
        """
        Copy array. Dask arrays are immutable, but preserve the subclass.
        """
        return DeisaArray(
            self.t,
            self.dask,
            self.name,
            self.chunks,
            dtype=self.dtype,
            meta=self,
            shape=self.shape,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        c = self.copy()
        memo[id(self)] = c
        return c

    @property
    def timestep(self) -> int:
        return self.t

    def to_hdf5(self, filename, datapath, **kwargs) -> None:
        """
        Save data to a HDF5 file (using HDF5 VDS).

        Parameters
        ----------
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.

        Notes
        -----
        This method creates files for each chunk, then links the
        files using HDF5 VDS. The chunk file is named as
        `.{filename}-{chunk_position}.h5`.
        """

        DeisaArray.__to_hdf5(filename, {datapath: self})

    @staticmethod
    def __chunk_fname(fname: str, dataset: str, block_id: tuple[int, ...] = ()):
        """
        Create the filename for a chunk.

        Parameters
        ----------
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.
        block_id : tuple[int, ...]
            Chunk position to create the file.

        Returns
        -------
        str
            Filename for the chunk of position block_id.
        """

        path = pathlib.Path(fname).expanduser().resolve()
        parents, name, suffix = path.parents[0], path.stem, path.suffix
        chunk_str = "-".join(map(str, block_id))

        # Hidden name for the chunk files
        new_name = "." + name + "-" + dataset + f"-{chunk_str}" + suffix

        return parents / new_name

    @staticmethod
    def __save_chunk(chunk: np.ndarray, fname: str, dataset: str, block_id: tuple[int, ...]) -> None:
        """
        Save one chunk to a individual hdf5 file.

        Parameters
        ----------
        chunk : np.ndarray
            Chunk to be stored.
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.
        block_id : tuple[int, ...]
            Chunk position, used to merge into a VDS.
        """

        import h5py

        filename = DeisaArray.__chunk_fname(fname, dataset, block_id)

        with h5py.File(filename, "w") as f:
            f.create_dataset(dataset, data=chunk)

    @staticmethod
    def __create_vds(
        fname: Union[str, pathlib.Path],
        dataset: str,
        chunk_shape: tuple[int, ...],
        data_shape: tuple[int, ...],
        nb_chunks_per_dim: tuple[int, ...],
        data_dtype: np.dtype,
    ) -> None:
        """
        Creates a VDS aggregating all chunk files.

        Parameters
        ----------
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.
        chunk_shape : tuple[int,...]
            Shape of the chunks, used to map the chunks into the VDS.4
        data_shape : tuple[int,...]
            Shape of the data.
        nb_chunks_per_dim : tuple[int,...]
            Number of chunks for each dimension
        data_dtype : np.dtype
            The numpy dtype of the data.
        """

        import h5py

        # Defines the layout of the VDS file (it's shape) as the same as the dask array
        layout = h5py.VirtualLayout(shape=data_shape, dtype=data_dtype)

        for block_id in np.ndindex(nb_chunks_per_dim):
            # Transform each chunk file into a source for the VDS (a file where it takes data from)
            # and define where the data for the each chunk file will be placed in the final VDS layout.
            name = DeisaArray.__chunk_fname(fname, dataset, block_id)
            vsource = h5py.VirtualSource(name, dataset, shape=chunk_shape)
            selection = tuple(slice(idx * size, (idx + 1) * size) for idx, size in zip(block_id, chunk_shape))

            layout[selection] = vsource

        # Use the append ("a") mode for saving, this enables the creation of different datasets in the same file.
        with h5py.File(fname, "a", libver="latest") as f:
            # Create the hdf5 dataset inside of the VDS file, the fillvalue sets that
            # if a file didn't exists yet or can't be found when the VDS was open, it
            # will fill the -1 value in the place corresponding to that file.
            f.create_virtual_dataset(dataset, layout, fillvalue=-1)

    @staticmethod
    def __to_hdf5(fname: str, sources: dict[str, "DeisaArray"]) -> None:
        """
        Save data to a HDF5 file (using HDF5 VDS).

        Parameters
        ----------
        fname : str
            The name of the final file where the data will be stored.
        sources : dict[str, DeisaArray]
            Dict mapping the datasets in final file to final files

        Notes
        -----
        This method creates files for each chunk, then links the
        files using HDF5 VDS. The chunk file is named as
        `.{filename}-{dataset}-{chunk_position}.h5`. This may cause
        errors with datasets with special characters in the name.
        """

        full_path = pathlib.Path(fname).expanduser().resolve()

        writing_tasks = []
        for dataset, deisa_array in sources.items():
            # dask.to_delayed() be able to have a reference to the chunks
            # and to be able to iterate through them.
            delayed_grid = deisa_array.to_delayed()

            for block_id in np.ndindex(delayed_grid.shape):
                chunk = delayed_grid[block_id]
                writing_tasks.append(
                    dask.delayed(DeisaArray.__save_chunk)(
                        chunk, fname=str(full_path), dataset=dataset, block_id=block_id
                    )
                )

            # Create the dataset for the current DeisaArray inside the VDS.
            # note that he VDS file can be created before having the actual
            # files for the chunks created.
            DeisaArray.__create_vds(
                full_path,
                dataset,
                deisa_array.chunksize,
                deisa_array.shape,
                deisa_array.numblocks,
                deisa_array.dtype,
            )

        dask.compute(*writing_tasks)


def _as_shape(value: Any, what: str) -> Tuple[int, ...]:
    """Normalize ``value`` into a non-empty tuple of positive ints, or raise a refusal for a violated invariant."""
    try:
        shape = tuple(int(size) for size in value)
    except (TypeError, ValueError):
        raise errors.ondemand_invariant("OndemandDeisaArray", f"{what} {value!r} is not a shape") from None
    if not shape or any(size <= 0 for size in shape):
        raise errors.ondemand_invariant("OndemandDeisaArray", f"{what} {shape} is not a non-empty shape")
    return shape


def _chunk_grid(shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    """Tiling of ``shape`` by ``chunk_shape``, i.e. the dask ``chunks`` a one-chunk-per-bridge array describes.

    Derived, never carried: the harness invariant is one chunk per bridge, so the chunk shape IS the per-bridge
    extent and the grid follows from the global shape. A global extent that is not a whole multiple of the chunk
    extent still tiles: the last chunk of that axis is the remainder.
    """
    grid = []
    for total, chunk in zip(shape, chunk_shape):
        count = -(-total // chunk)  # ceil division
        grid.append((chunk,) * (count - 1) + (total - chunk * (count - 1),))
    return tuple(grid)


class _Metadata(NamedTuple):
    """The validated, derived description of one delivered array (what both the constructor and the factory need)."""

    name: str
    timestep: int
    shape: Tuple[int, ...]
    dtype: np.dtype
    chunk_shape: Tuple[int, ...]
    bridge_count: int
    chunks: Tuple[Tuple[int, ...], ...]


def _normalized_metadata(
    name: str,
    timestep: int,
    shape: Sequence[int],
    dtype: Any,
    chunk_shape: Optional[Sequence[int]],
    bridge_count: Optional[int],
) -> _Metadata:
    """Validate one array's metadata and derive the chunk grid and the bridge count from it.

    Both the constructor and :meth:`OndemandDeisaArray.from_metadata` read the array's identity from here, so the
    invariants (a non-empty shape, a chunk that fits inside it, a declared bridge count that agrees with them) live in
    exactly one place.
    """
    global_shape = _as_shape(shape, "the global shape")
    resolved_dtype = np.dtype(dtype)
    resolved_chunk_shape = global_shape if chunk_shape is None else _as_shape(chunk_shape, "the chunk shape")
    if len(resolved_chunk_shape) != len(global_shape):
        raise errors.ondemand_invariant(
            "OndemandDeisaArray",
            f"the chunk shape {resolved_chunk_shape} and the global shape {global_shape} have different ranks",
        )
    if any(chunk > size for chunk, size in zip(resolved_chunk_shape, global_shape)):
        raise errors.ondemand_invariant(
            "OndemandDeisaArray",
            f"the chunk shape {resolved_chunk_shape} does not fit inside the global shape {global_shape}",
        )
    # Per-data-axis extent of the MPI chunk grid: the harness invariant that the cart dims map one-to-one onto the
    # array's data axes, so the product IS the number of bridges holding a chunk of this array.
    derived_count = int(np.prod([max(1, size // chunk) for chunk, size in zip(resolved_chunk_shape, global_shape)]))
    resolved_name = str(name)
    if bridge_count is not None and int(bridge_count) != derived_count:
        raise errors.ondemand_invariant(
            "OndemandDeisaArray",
            f"the metadata implies {derived_count} bridge(s) for array {resolved_name!r} "
            f"(global shape {global_shape}, chunk shape {resolved_chunk_shape}) but {bridge_count} was declared",
        )
    return _Metadata(
        name=resolved_name,
        timestep=int(timestep),
        shape=global_shape,
        dtype=resolved_dtype,
        chunk_shape=resolved_chunk_shape,
        bridge_count=derived_count,
        chunks=_chunk_grid(global_shape, resolved_chunk_shape),
    )


class _NoGraph(Mapping):
    """The empty graph of a data-free operand: every read of it is a refusal, never a dict.

    A data-free array must not be usable as an input to a foreign dask computation: the callback builds a graph,
    ``compute()`` gathers the field onto the analytics host, and nothing in the delivered metadata shows the gather.
    Dask's own entry points (``da.sum(arr)``, ``da.tensordot(arr, arr)``) reach the graph before any operator of ours
    does, so the graph itself is what refuses -- the operand ``isinstance`` an array, it just has nothing behind it.

    It is a ``Mapping`` because that is what ``dask.array.Array`` stores, so every consumer's ``dict``/``merge``/
    ``.layers`` step lands on one of the three methods below.
    """

    __slots__ = ("_symbol",)

    def __init__(self, symbol: str) -> None:
        self._symbol = str(symbol)

    def _refuse(self) -> Any:
        raise errors.off_surface(
            f"the dask graph of {self._symbol}",
            "the operand carries no graph: its field is never delivered, so any dask computation built over it "
            "would gather the whole registered array",
        )

    def __getitem__(self, key: Any) -> Any:
        self._refuse()

    def __iter__(self) -> Iterator[Any]:
        self._refuse()

    def __len__(self) -> int:
        self._refuse()

    def __getattr__(self, name: str) -> Any:
        # ``.layers``/``.merge``/``.to_dict``: every HighLevelGraph entry point, refused the same way.
        if name.startswith("_"):
            raise AttributeError(name)
        self._refuse()


class OndemandDeisaArray(LazyOperand, DeisaArray):
    """One registered array as a ``precompute=True`` callback sees it at one timestep.

    It is a :class:`DeisaArray` -- ``isinstance(x, da.Array)`` and ``isinstance(x, DeisaArray)`` both hold, so a
    callback written against the public contract cannot tell the delivery modes apart -- and at the same time a
    :class:`~deisa.dask.ondemand_expr.LazyOperand`, which is what supplies the closed operator surface, the
    provenance rules and the materialization guards. The lazy surface comes FIRST in the MRO on purpose: dask's own
    ``__array__``/``__iter__``/``__bool__``/``__len__``/``__getitem__`` would otherwise compute the array.

    The dask graph is EMPTY by construction and refuses to be read (:class:`_NoGraph`): there is no chunk on the
    analytics host to put in it. Every dask ``Array`` method that would build a graph over that emptiness is refused
    (see :data:`_REFUSED_ARRAY_METHODS`) instead of returning an array that dies later with a missing-key error.

    - ``:param name:`` Registered array name (e.g. ``"f"``). Its ``name`` is half of the root, so a reduction can never
      be served by another array's chunk.
    - ``:param timestep:`` Simulation timestep this instance was delivered for. The other half of the root: two
      operands of the same array at different timesteps are two different roots and cannot be combined.
    - ``:param shape:`` GLOBAL shape of the array (all bridges), e.g. ``(8, 8)``.
    - ``:param dtype:`` Element dtype of the array.
    - ``:param chunk_shape:`` Shape of ONE bridge's chunk (the harness invariant is one chunk per bridge). Defaults to
      the global shape, i.e. a single-bridge array.
    - ``:param bridge_count:`` Number of bridges that own a chunk. Auto-detected from ``shape`` and ``chunk_shape``
      when omitted; when given it is CROSS-CHECKED against them, because a caller-supplied count that disagrees with
      the metadata would make every reduction's combine grid wrong.
    - ``:param dispatcher:`` The compute boundary that delivers this operand's reductions, attached by the delivery
      path. Left ``None`` by an operand built outside that path, whose reductions then refuse instead of gathering.
    """

    __slots__ = ()

    def __new__(
        cls,
        name: str,
        timestep: int,
        shape: Sequence[int],
        dtype: Any = np.float64,
        chunk_shape: Optional[Sequence[int]] = None,
        bridge_count: Optional[int] = None,
        dispatcher: Any = None,
    ):
        # The K9-certified class swap: build a genuine DeisaArray first, so dask's own allocation runs unaltered
        # (``dask.array.Array.__new__`` reads ``self.ndim``/``self.shape`` from dask's chunk bookkeeping, which the lazy
        # surface would answer before its state exists), then transplant the operand class onto it. Both classes have
        # the same instance layout, so the swap is the one dask itself performs when a derived subclass is dropped.
        metadata = _normalized_metadata(name, timestep, shape, dtype, chunk_shape, bridge_count)
        operand = DeisaArray(
            metadata.timestep,
            {},
            metadata.name,
            metadata.chunks,
            dtype=metadata.dtype,
            meta=None,
            shape=metadata.shape,
        )
        operand.__class__ = cls
        return operand

    def __init__(
        self,
        name: str,
        timestep: int,
        shape: Sequence[int],
        dtype: Any = np.float64,
        chunk_shape: Optional[Sequence[int]] = None,
        bridge_count: Optional[int] = None,
        dispatcher: Any = None,
    ) -> None:
        metadata = _normalized_metadata(name, timestep, shape, dtype, chunk_shape, bridge_count)
        self._shape = metadata.shape
        self._dtype = metadata.dtype
        self.chunk_shape = metadata.chunk_shape
        self.bridge_count = metadata.bridge_count
        self._root = Root(array_name=metadata.name, timestep=metadata.timestep)
        self._program = _CHUNK_PROGRAM
        self._dispatcher = dispatcher
        # Replace the empty HighLevelGraph ``Array.__new__`` wrapped around it: the operand has no task to run, and
        # every dask computation that reaches for one must refuse instead of computing a hole.
        self.dask = _NoGraph(self._symbol())

    @classmethod
    def from_metadata(
        cls,
        name: str,
        timestep: int,
        metadata: Mapping[str, Any],
        bridge_count: Optional[int] = None,
        dispatcher: Any = None,
    ) -> "OndemandDeisaArray":
        """Build the operand from the array metadata the handshake already exposes.

        The delivery site (the topic handler, the bridge's aggregates step) has the metadata dict, not a shape:
        reading it here keeps the key spelling in one place instead of at every call site.

        - ``:param name:`` Registered array name.
        - ``:param timestep:`` Timestep the operand is delivered for.
        - ``:param metadata:`` Entry of the handshake's ``arrays_metadata`` for this array (``global_shape``,
          ``chunk_shape``, optional ``dtype``).
        - ``:param bridge_count:`` Optional cross-check, as in the constructor.
        - ``:param dispatcher:`` The compute boundary to attach, as in the constructor.
        """
        global_shape = metadata.get("global_shape")
        if not global_shape:
            raise errors.ondemand_invariant(
                "OndemandDeisaArray.from_metadata", f"the metadata for array {name!r} carries no ``global_shape``"
            )
        return cls(
            name=name,
            timestep=timestep,
            shape=global_shape,
            dtype=metadata.get("dtype", np.float64),
            chunk_shape=metadata.get("chunk_shape"),
            bridge_count=bridge_count,
            dispatcher=dispatcher,
        )

    def _symbol(self) -> str:
        # Verbatim from the pre-K13C carrier: the refusal messages are contract (guardrail K13C), so the delivered
        # operand keeps the wording the suite asserts even though the class is now a DeisaArray.
        return f"the collection {self.name!r} at timestep {self.timestep}"

    @property
    def name(self) -> str:
        """The registered array name: write-mutable metadata, as the delivery contract always treated it.

        Dask's own ``Array.name`` setter refuses a direct write because the name keys a task graph; this operand's
        graph is :class:`_NoGraph` (nothing keys on it), so the pre-K13C ``DeisaCollection`` semantics survive: a
        callback may rename its instance without touching the dask bookkeeping.
        """
        try:
            return self._deisa_name
        except AttributeError:
            return super(OndemandDeisaArray, type(self)).name.fget(self)

    @name.setter
    def name(self, value: Any) -> None:
        self._deisa_name = str(value)

    # -- dask protocols that would otherwise compute or build a graph --------------------------------------------------
    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Refuse a numpy ufunc applied to the operand: ``np.sqrt(arr)`` is a raw read of the field."""
        raise errors.numpy_materialization(self._symbol())

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        """Refuse a numpy function applied to the operand: ``np.linalg.norm(arr)`` would gather the field."""
        raise errors.numpy_materialization(self._symbol())

    @property
    def T(self) -> Any:
        """Refused: a transpose is a dask graph over chunks the analytics host does not have."""
        raise errors.off_surface(".T", "the attribute is not part of the closed DEISA operator surface")

    def __repr__(self) -> str:
        return (
            f"OndemandDeisaArray(name={self.name!r}, timestep={self.timestep}, shape={self._shape}, "
            f"dtype={self._dtype}, chunk_shape={self.chunk_shape}, bridge_count={self.bridge_count})"
        )


# The rest of the dask ``Array`` surface: every entry builds a graph (or moves data) over a field that was never
# delivered, so each one is refused with the closed-surface message the suite asserts. Metadata properties
# (``shape``/``dtype``/``ndim``/``size``/``nbytes``/``chunks``/``chunksize``/``numblocks``/``name``/``itemsize``) stay
# readable -- they describe the registered array and are what a callback is allowed to size itself from.
_REFUSED_ARRAY_METHODS: Tuple[str, ...] = (
    "argtopk",
    "astype",
    "blocks",
    "compute_chunk_sizes",
    "conj",
    "copy",
    "cumprod",
    "cumsum",
    "dot",
    "flatten",
    "map_blocks",
    "map_overlap",
    "moment",
    "nonzero",
    "partitions",
    "persist",
    "ravel",
    "rechunk",
    "repeat",
    "reshape",
    "round",
    "shuffle",
    "squeeze",
    "store",
    "swapaxes",
    "to_backend",
    "to_dask_dataframe",
    "to_delayed",
    "to_svg",
    "to_tiledb",
    "to_zarr",
    "topk",
    "trace",
    "transpose",
    "view",
    "vindex",
    "visualize",
)


class _RefusedArrayMethod:
    """Descriptor standing in for a dask ``Array`` method a data-free operand must not expose.

    The pre-K13C carrier was NOT a ``da.Array``, so ``operand.reshape`` missed the class and landed on
    :meth:`LazyOperand.__getattr__`, which refuses at ATTRIBUTE ACCESS. The operand is a ``da.Array`` now, so dask's
    own method really is on the class and would win; this descriptor keeps the refusal at the same point -- access, not
    the later call -- by refusing through that same ``__getattr__`` path, so the wording stays byte-identical.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        if instance is None:
            return self
        # Delegated (not re-formatted) so the closed-surface message lives in exactly one place.
        return LazyOperand.__getattr__(instance, self._name)


for _refused_name in _REFUSED_ARRAY_METHODS:
    setattr(OndemandDeisaArray, _refused_name, _RefusedArrayMethod(_refused_name))
del _refused_name

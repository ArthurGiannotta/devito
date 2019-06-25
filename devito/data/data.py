from collections import Iterable
from functools import wraps

import numpy as np

from devito.data.allocators import ALLOC_FLAT
from devito.parameters import configuration
from devito.tools import Tag, as_tuple, is_integer

__all__ = ['Data']


class Data(np.ndarray):

    """
    A numpy.ndarray supporting distributed Dimensions.

    Parameters
    ----------
    shape : tuple of ints
        Shape of created array.
    dtype : numpy.dtype
        The data type of the raw data.
    decomposition : tuple of Decomposition, optional
        The data decomposition, for each dimension.
    modulo : tuple of bool, optional
        If the i-th entry is True, then the i-th array dimension uses modulo indexing.
    allocator : MemoryAllocator, optional
        Used to allocate memory. Defaults to ``ALLOC_FLAT``.

    Notes
    -----
    NumPy array subclassing is described at: ::

        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    Any view or copy created from ``self``, for instance via a slice operation
    or a universal function ("ufunc" in NumPy jargon), will still be of type
    Data.
    """

    def __new__(cls, shape, dtype, decomposition=None, modulo=None, allocator=ALLOC_FLAT,
                distributor=None):
        assert len(shape) == len(modulo)
        ndarray, memfree_args = allocator.alloc(shape, dtype)
        obj = np.asarray(ndarray).view(cls)
        obj._allocator = allocator
        obj._memfree_args = memfree_args
        obj._decomposition = decomposition or (None,)*len(shape)
        obj._modulo = modulo or (False,)*len(shape)
        obj._distributor = distributor

        # This cannot be a property, as Data objects constructed from this
        # object might not have any `decomposition`, but they would still be
        # distributed. Hence, in `__array_finalize__` we must copy this value
        obj._is_distributed = any(i is not None for i in obj._decomposition)

        # Saves the last index used in `__getitem__`. This allows `__array_finalize__`
        # to reconstruct information about the computed view (e.g., `decomposition`)
        obj._index_stash = None

        # Sanity check -- A Dimension can't be at the same time modulo-iterated
        # and MPI-distributed
        assert all(i is None for i, j in zip(obj._decomposition, obj._modulo)
                   if j is True)

        return obj

    def __del__(self):
        if self._memfree_args is None:
            return
        self._allocator.free(*self._memfree_args)
        self._memfree_args = None

    def __array_finalize__(self, obj):
        # `self` is the newly created object
        # `obj` is the object from which `self` was created
        if obj is None:
            # `self` was created through __new__()
            return

        self._distributor = None
        self._index_stash = None

        # Views or references created via operations on `obj` do not get an
        # explicit reference to the underlying data (`_memfree_args`). This makes sure
        # that only one object (the "root" Data) will free the C-allocated memory
        self._memfree_args = None

        if type(obj) != Data:
            # Definitely from view casting
            self._is_distributed = False
            self._modulo = tuple(False for i in range(self.ndim))
            self._decomposition = (None,)*self.ndim
        elif obj._index_stash is not None:
            # From `__getitem__`
            self._is_distributed = obj._is_distributed
            self._distributor = obj._distributor
            glb_idx = obj._normalize_index(obj._index_stash)
            self._modulo = tuple(m for i, m in zip(glb_idx, obj._modulo)
                                 if not is_integer(i))
            decomposition = []
            for i, dec in zip(glb_idx, obj._decomposition):
                if is_integer(i):
                    continue
                elif dec is None:
                    decomposition.append(None)
                else:
                    decomposition.append(dec.reshape(i))
            self._decomposition = tuple(decomposition)
        else:
            self._is_distributed = obj._is_distributed
            self._distributor = obj._distributor
            if self.ndim == obj.ndim:
                # E.g., from a ufunc, such as `np.add`
                self._modulo = obj._modulo
                self._decomposition = obj._decomposition
            else:
                # E.g., from a reduction operation such as `np.mean` or `np.all`
                self._modulo = tuple(False for i in range(self.ndim))
                self._decomposition = (None,)*self.ndim

    @property
    def _local(self):
        """A view of ``self`` with global indexing disabled."""
        ret = self.view()
        ret._is_distributed = False
        return ret

    def _global(self, glb_idx, decomposition):
        """A "global" view of ``self`` over a given Decomposition."""
        if self._is_distributed:
            raise ValueError("Cannot derive a decomposed view from a decomposed Data")
        if len(decomposition) != self.ndim:
            raise ValueError("`decomposition` should have ndim=%d entries" % self.ndim)
        ret = self[glb_idx]
        ret._decomposition = decomposition
        ret._is_distributed = any(i is not None for i in decomposition)
        return ret

    def check_slicing(func):
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            glb_idx = args[0]
            if data._is_mpi_distributed:
                for i in as_tuple(glb_idx):
                    if isinstance(i, slice) and i.step is not None and i.step < 0:
                        mpi_slicing = True
                        break
                    else:
                        mpi_slicing = False
            else:
                mpi_slicing = False
            kwargs['mpi_slicing'] = mpi_slicing
            return func(data, *args, **kwargs)
        return wrapper

    @property
    def _is_mpi_distributed(self):
        return self._is_distributed and configuration['mpi']

    def __repr__(self):
        return super(Data, self._local).__repr__()

    @check_slicing
    def __getitem__(self, glb_idx, mpi_slicing=False):
        loc_idx = self._convert_index(glb_idx)
        if mpi_slicing:
            # FIXME: Still need to deal with the case when len(loc_idx) < len(local_val.shape)
            # Retrieve the pertinent local data prior to mpi send/receive operations
            loc_data_idx = []
            for i in as_tuple(loc_idx):
                # NOTE: Can probably remove 'and i.step is not None'
                if isinstance(i, slice) and i.step is not None and i.step < 0:
                    if i.stop is None:
                        loc_data_idx.append(slice(0, i.start+1, -i.step))
                    else:
                        loc_data_idx.append(slice(i.stop+1, i.start+1, -i.step))
                elif is_integer(i):
                    loc_data_idx.append(slice(i, i+1, 1))
                else:
                    loc_data_idx.append(i)
            loc_data_idx = as_tuple(loc_data_idx)

            # FIXME: local_val currently returning the wrong decompositin (general problem
            # and not related to this branch)
            local_val = super(Data, self).__getitem__(loc_data_idx)

            # NOTE: If 'local_val.size == 0' return on a specific rank we can probably
            # return now. Double check.
            #if local_val.size == 0:
                #return local_val

            rank = self._distributor.myrank
            comm = self._distributor.comm
            nprocs = self._distributor.nprocs
            topology = self._distributor.topology
            # Produce the 'rank' matrix
            rank_mat = np.arange(nprocs).reshape(topology)
            # Gather data structures from all ranks in order to produce the
            # relevant mappings. Mask ranks with no data.
            # FIXME: Much of the below can be combined
            rank_coords = self._distributor.all_coords
            dat_struct = []
            dat_len = np.zeros(topology, dtype=tuple)
            mask = np.zeros(nprocs, dtype=np.int32)
            for j in range(nprocs):
                    dat_len[rank_coords[j]] = comm.bcast(local_val.shape, root=j)
                    dat_struct.append(dat_len[rank_coords[j]])
                    #dat_struct.append(comm.bcast(local_val.shape, root=j))
                    if any(k == 0 for k in dat_struct[j]):
                        mask[j] = 1
            for i in rank_coords:
                if any([j == 0 for j in dat_len[i]]):
                    dat_len[i] = as_tuple([0]*len(i))
            #for i in dat_len:
                #if any([j == 0 for j in i]):
                    #i = as_tuple([0]*len(i))
            dat_len_cum = np.zeros(topology, dtype=tuple)
            for i in range(nprocs):
                my_coords = rank_coords[i]
                if i == 0:
                    dat_len_cum[my_coords] = dat_len[my_coords]
                    continue
                mod_coords = []
                for j in range(len(my_coords)):
                    shift_coords = list(my_coords)
                    shift_coords[j] -= 1
                    mod_coords.append(as_tuple(shift_coords))
                mod_coords = as_tuple(mod_coords)
                #for j1, j2 in zip(my_coords, mod_coords):
                    #n_slice = []
                n_dat = []
                for j in range(len(my_coords)):
                    if mod_coords[j][j] < 0:
                        c_dat = dat_len[my_coords][j]
                        n_dat.append(c_dat)
                    else:
                        c_dat = dat_len[my_coords][j]
                        p_dat = dat_len[mod_coords[j]][j]
                        n_dat.append(c_dat+p_dat)
                dat_len_cum[my_coords] = as_tuple(n_dat)
            # This 'transform' will be required to produce the required maps
            # NOTE: Doudble check the 'else 0' is robust.
            transform = []
            for i in as_tuple(loc_idx):
                if isinstance(i, slice):
                    if i.step is not None:
                        transform.append(slice(None, None, np.sign(i.step)))
                    else:
                        transform.append(slice(None, None, None))
                else:
                    transform.append(0)
            transform = as_tuple(transform)
            # Maksed rank matrices
            m_rank_mat = np.ma.masked_array(rank_mat, mask=mask.reshape(topology))
            m_rank_mat[None, ~m_rank_mat.mask] = m_rank_mat[None, ~m_rank_mat.mask][transform]
            m_rank_mat_t = m_rank_mat.reshape(nprocs)

            # FIXME: Better ways of doing this
            global_size = []
            for i, j in zip(glb_idx, self._distributor._glb_shape):
                if isinstance(i, slice):
                    if i.start is None and i.step is not None and i.step < 0:
                        start = j-1
                    elif i.start is None:
                        start = 0
                    else:
                        start = i.start
                    if i.stop is None and i.step is not None and i.step < 0:
                        stop = -1
                    elif i.stop is None:
                        stop = j
                    else:
                        stop = i.stop
                    global_size.append(abs(start-stop))
                else:
                    global_size.append(1)
            global_size = as_tuple(global_size)

            tups = np.zeros(global_size, dtype=tuple)
            global_si = np.zeros(global_size, dtype=tuple)
            it =  np.nditer(tups, flags=['refs_ok', 'multi_index'])
            while not it.finished:
                index = it.multi_index
                tups[index] = index
                it.iternext()
            global_si[:] = tups[transform]

            # create the 'rank' slices
            rank_slice = []
            for j in dat_struct:
                this_rank = []
                for k in j:
                    this_rank.append(slice(0, k, 1))
                rank_slice.append(this_rank)
            # Normalize the slices:
            # FIXME: USe zip
            n_rank_slice = []
            for i in range(len(rank_slice)):
                my_coords = rank_coords[i]
                if any([j.stop == j.start for j in rank_slice[i]]):
                    n_rank_slice.append(as_tuple([None]*len(rank_slice[i])))
                    continue
                if i == 0:
                    n_rank_slice.append(as_tuple(rank_slice[i]))
                    continue
                mod_coords = []
                for j in range(len(my_coords)):
                    shift_coords = list(my_coords)
                    shift_coords[j] -= 1
                    mod_coords.append(as_tuple(shift_coords))
                mod_coords = as_tuple(mod_coords)
                n_slice = []
                for j in range(len(my_coords)):
                    if mod_coords[j][j] < 0:
                        start = 0
                        stop = dat_len_cum[my_coords][j]
                    else:
                        start = dat_len_cum[mod_coords[j]][j]
                        stop = dat_len_cum[my_coords][j]
                    n_slice.append(slice(start, stop, 1))
                n_rank_slice.append(as_tuple(n_slice))
            #for i in range(0, len(rank_slice)):
                #if any([j.stop == j.start for j in rank_slice[i]]):
                    #n_rank_slice.append(as_tuple([None]*len(rank_slice[i])))
                    #continue
                #if i == 0:
                    #n_rank_slice.append(as_tuple(rank_slice[i]))
                    #continue
                #my_coords = rank_coords[j]
                #mod_coords = []
                #for j in range(len(my_coords)):
                    #shift_coords = list(my_coords)
                    #shift_coords[j] -= 1
                    #mod_coords.append(as_tuple(shift_coords))
                #mod_coords = as_tuple(mod_coords)
                #for j1, j2 in zip(my_coords, mod_coords):
                    #n_slice = []
                ##for in zip(, rank_coords[i], rank_coords[i-1])
                ##for j1, j2, k1, k2 in zip(rank_slice[i], n_rank_slice[i-1], rank_coords[i], rank_coords[i-1]):
                    ##shift = k1 - k2
                    ##if shift == 1 and j2 is not None:
                        ##start = j2.stop
                        ##stop = start + j1.stop
                        ##n_slice.append(slice(start, stop, 1))
                    ##elif shift <= 0 or j2 is None:
                        ##n_slice.append(j1)
                    ##else:
                        ##raise ValueError("Unexpected rank shift")
                #n_rank_slice.append(as_tuple(n_slice))
            n_rank_slice = as_tuple(n_rank_slice)
            #from IPython import embed; embed()

            # we know how to modify the slices from using m_rank_mat:
            # Now fill each elements owner:
            owners = np.zeros(global_size, dtype=np.int32)
            send = np.zeros(global_size, dtype=np.int32)
            for i in range(len(n_rank_slice)):
                if any([j == None for j in n_rank_slice[i]]):
                    continue
                else:
                    owners[n_rank_slice[i]] = i
            send[:] = owners[transform]

            # local_indices
            local_si = np.zeros(global_size, dtype=tuple)
            ita =  np.nditer(local_si, flags=['refs_ok', 'multi_index'])
            while not ita.finished:
                index = ita.multi_index
                owner = owners[index]
                my_slice = n_rank_slice[owner]
                rnorm_index = []
                for j, k in zip(my_slice, index):
                    rnorm_index.append(k-j.start)
                local_si[index] = as_tuple(rnorm_index)
                ita.iternext()

            # copy then overwrite?
            # FIXME: Needs to be type Data
            retval = Data(local_val.shape, local_val.dtype.type,
                          decomposition=local_val._decomposition, modulo=local_val._modulo,
                          distributor=local_val._distributor)
            it2 =  np.nditer(owners, flags=['refs_ok', 'multi_index'])
            #from IPython import embed; embed()
            while not it2.finished:
                index = it2.multi_index
                if rank == owners[index] and rank == send[index]:
                    loc_ind = local_si[index]
                    send_ind = local_si[global_si[index]]
                    retval.data[send_ind] = local_val.data[loc_ind]
                elif rank == owners[index]:
                    loc_ind = local_si[index]
                    send_rank = send[index]
                    send_ind = global_si[index]
                    send_val = local_val.data[loc_ind]
                    reqs = comm.isend([send_ind, send_val], dest=send_rank)
                    reqs.wait()
                elif rank == send[index]:
                    recval = comm.irecv(source=owners[index])
                    local_dat = recval.wait()
                    loc_ind = local_si[local_dat[0]]
                    retval.data[loc_ind] = local_dat[1]
                else:
                    pass
                it2.iternext()
            return retval
        elif loc_idx is NONLOCAL:
            # Caller expects a scalar. However, `glb_idx` doesn't belong to
            # self's data partition, so None is returned
            return None
        else:
            self._index_stash = glb_idx
            retval = super(Data, self).__getitem__(loc_idx)
            self._index_stash = None
            return retval

    @check_slicing
    def __setitem__(self, glb_idx, val, mpi_slicing=False):
        loc_idx = self._convert_index(glb_idx)
        if loc_idx is NONLOCAL:
            # no-op
            return
        elif np.isscalar(val):
            if index_is_basic(loc_idx):
                # Won't go through `__getitem__` as it's basic indexing mode,
                # so we should just propage `loc_idx`
                super(Data, self).__setitem__(loc_idx, val)
            else:
                super(Data, self).__setitem__(glb_idx, val)
        elif isinstance(val, Data) and val._is_distributed:
            if mpi_slicing:
                # `val` is decomposed, `self` is decomposed -> local set
                # FIXME: need to fix the new decomp for RHS's such as f.data[-2:, -2:]
                val_idx = as_tuple([slice(i.glb_min, i.glb_max+1, 1) for
                                    i in val._decomposition])
                idx = self._set_global_idx(val, glb_idx, val_idx)
                comm = self._distributor.comm
                nprocs = self._distributor.nprocs
                # Prepare global lists:
                data_global = []
                idx_global = []
                for j in range(nprocs):
                    data_global.append(comm.bcast(np.array(val), root=j))
                    idx_global.append(comm.bcast(idx, root=j))
                # Set the data:
                for j in range(nprocs):
                    skip = any(i is None for i in idx_global[j])
                    if not skip:
                        loc_idx_new = self._convert_index(idx_global[j])
                        if loc_idx_new is NONLOCAL:
                            return
                        else:
                            self.__setitem__(idx_global[j], data_global[j])
            elif self._is_distributed:
                # `val` is decomposed, `self` is decomposed -> local set
                super(Data, self).__setitem__(glb_idx, val)
            else:
                # `val` is decomposed, `self` is replicated -> gatherall-like
                raise NotImplementedError
        elif isinstance(val, np.ndarray):
            if self._is_distributed:
                # `val` is replicated, `self` is decomposed -> `val` gets decomposed
                glb_idx = self._normalize_index(glb_idx)
                glb_idx, val = self._process_args(glb_idx, val)
                val_idx = [index_dist_to_repl(i, dec) for i, dec in
                           zip(glb_idx, self._decomposition)]
                if NONLOCAL in val_idx:
                    # no-op
                    return
                val_idx = tuple([i for i in val_idx if i is not PROJECTED])
                # NumPy broadcasting note:
                # When operating on two arrays, NumPy compares their shapes
                # element-wise. It starts with the trailing dimensions, and works
                # its way forward. Two dimensions are compatible when
                # * they are equal, or
                # * one of them is 1
                # Conceptually, below we apply the same rule
                val_idx = val_idx[len(val_idx)-val.ndim:]
                val = val[val_idx]
            else:
                # `val` is replicated`, `self` is replicated -> plain ndarray.__setitem__
                pass
            super(Data, self).__setitem__(glb_idx, val)
        elif isinstance(val, Iterable):
            if self._is_mpi_distributed:
                raise NotImplementedError("With MPI data can only be set "
                                          "via scalars or numpy arrays")
            super(Data, self).__setitem__(glb_idx, val)
        else:
            raise ValueError("Cannot insert obj of type `%s` into a Data" % type(val))

    def _normalize_index(self, idx):
        if isinstance(idx, np.ndarray):
            # Advanced indexing mode
            return (idx,)
        else:
            idx = as_tuple(idx)
            if any(i is Ellipsis for i in idx):
                # Explicitly replace the Ellipsis
                items = (slice(None),)*(self.ndim - len(idx) + 1)
                items = idx[:idx.index(Ellipsis)] + items + idx[idx.index(Ellipsis)+1:]
            else:
                items = idx + (slice(None),)*(self.ndim - len(idx))
            # Normalize slice steps:
            processed = [slice(i.start, i.stop, 1) if
                         (isinstance(i, slice) and i.step is None)
                         else i for i in items]
            return as_tuple(processed)

    def _process_args(self, idx, val):
        if any(isinstance(i, slice) for i in idx):
            processed = []
            op = []
            for j in idx:
                if isinstance(j, slice) and j.step < 0:
                    if j.start is None:
                        stop = None
                    else:
                        stop = j.start + 1
                    if j.stop is None:
                        start = None
                    else:
                        start = j.stop + 1
                    processed.append(slice(start, stop, -j.step))
                    op.append(slice(None, None, j.step))
                else:
                    processed.append(j)
            return as_tuple(processed), val[as_tuple(op)]
        else:
            return idx, val

    def _convert_index(self, glb_idx):
        glb_idx = self._normalize_index(glb_idx)
        if len(glb_idx) > self.ndim:
            # Maybe user code is trying to add a new axis (see np.newaxis),
            # so the resulting array will be higher dimensional than `self`,
            if self._is_mpi_distributed:
                raise ValueError("Cannot increase dimensionality of MPI-distributed Data")
            else:
                # As by specification, we are forced to ignore modulo indexing
                return glb_idx

        loc_idx = []
        for i, s, mod, dec in zip(glb_idx, self.shape, self._modulo, self._decomposition):
            if mod is True:
                # Need to wrap index based on modulo
                v = index_apply_modulo(i, s)
            elif self._is_distributed is True and dec is not None:
                # Need to convert the user-provided global indices into local indices.
                # Obviously this will have no effect if MPI is not used
                try:
                    v = index_glb_to_loc(i, dec)
                except TypeError:
                    if self._is_mpi_distributed:
                        raise NotImplementedError("Unsupported advanced indexing with "
                                                  "MPI-distributed Data")
                    v = i
            else:
                v = i

            # Handle non-local, yet globally legal, indices
            v = index_handle_oob(v)

            loc_idx.append(v)

        # Deal with NONLOCAL accesses
        if NONLOCAL in loc_idx:
            if len(loc_idx) == self.ndim and index_is_basic(loc_idx):
                # Caller expecting a scalar -- it will eventually get None
                loc_idx = [NONLOCAL]
            else:
                # Caller expecting an array -- it will eventually get a 0-length array
                loc_idx = [slice(-1, -2) if i is NONLOCAL else i for i in loc_idx]

        return loc_idx[0] if len(loc_idx) == 1 else tuple(loc_idx)

    def _set_global_idx(self, val, sl1, sl2):
        data_loc_idx = val._convert_index(sl2)
        data_global_idx = []
        for i in range(len(sl2)):
            if not val._decomposition[i].loc_empty:
                data_global_idx.append(
                    val._decomposition[i].convert_index_global(data_loc_idx[i]))
            else:
                data_global_idx.append(None)
        # work out bits of sl1 data_global_idx correspond to
        norms = []
        # FIXME: sl1 and sl2 should probably be normalised prior to this point:
        # Maybe create an 'as_slice' function?
        for i, j in zip(as_tuple(sl1), as_tuple(sl2)):
            if isinstance(i, slice):
                # Don't need 'norms.append(i.start-j.start)'?
                norms.append(i.start)
            elif i is None:
                norms.append(0)
            else:
                norms.append(i)
        mapped_idx = []
        for i, j in zip(data_global_idx, norms):
            if i is not None:
                mapped_idx.append(slice(i.start+j, i.stop+j, i.step))
            else:
                mapped_idx.append(None)
        return as_tuple(mapped_idx)

    def reset(self):
        """Set all Data entries to 0."""
        self[:] = 0.0


class Index(Tag):
    pass
NONLOCAL = Index('nonlocal')  # noqa
PROJECTED = Index('projected')


def index_is_basic(idx):
    if is_integer(idx):
        return True
    elif isinstance(idx, (slice, np.ndarray)):
        return False
    else:
        return all(is_integer(i) or (i is NONLOCAL) for i in idx)


def index_apply_modulo(idx, modulo):
    if is_integer(idx):
        return idx % modulo
    elif isinstance(idx, slice):
        if idx.start is None:
            start = idx.start
        elif idx.start >= 0:
            start = idx.start % modulo
        else:
            start = -(idx.start % modulo)
        if idx.stop is None:
            stop = idx.stop
        elif idx.stop >= 0:
            stop = idx.stop % (modulo + 1)
        else:
            stop = -(idx.stop % (modulo + 1))
        return slice(start, stop, idx.step)
    elif isinstance(idx, (tuple, list)):
        return [i % modulo for i in idx]
    elif isinstance(idx, np.ndarray):
        return idx
    else:
        raise ValueError("Cannot apply modulo to index of type `%s`" % type(idx))


def index_dist_to_repl(idx, decomposition):
    """Convert a distributed array index into a replicated array index."""
    if decomposition is None:
        return PROJECTED if is_integer(idx) else slice(None)

    # Derive shift value
    if isinstance(idx, slice):
        if idx.step is None or idx.step >= 0:
            value = idx.start
        else:
            value = idx.stop
    else:
        value = idx
    if value is None:
        value = 0
    elif not is_integer(value):
        raise ValueError("Cannot derive shift value from type `%s`" % type(value))

    # Convert into absolute local index
    idx = decomposition.convert_index(idx, rel=False)

    if is_integer(idx):
        return PROJECTED
    elif idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i - value for i in idx]
    elif isinstance(idx, np.ndarray):
        return idx - value
    elif isinstance(idx, slice):
        if idx.step is not None and idx.step < 0:
            if idx.stop is None:
                return slice(idx.start - value, None, idx.step)
        return slice(idx.start - value, idx.stop - value, idx.step)
    else:
        raise ValueError("Cannot apply shift to type `%s`" % type(idx))


def index_glb_to_loc(idx, decomposition):
    """Convert a global index into a local index."""
    if is_integer(idx) or isinstance(idx, slice):
        return decomposition(idx)
    elif isinstance(idx, (tuple, list)):
        return [decomposition(i) for i in idx]
    elif isinstance(idx, np.ndarray):
        return np.vectorize(lambda i: decomposition(i))(idx)
    else:
        raise ValueError("Cannot convert global index of type `%s` into a local index"
                         % type(idx))


def index_handle_oob(idx):
    # distributed.convert_index returns None when the index is globally
    # legal, but out-of-bounds for the calling MPI rank
    if idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i for i in idx if i is not None]
    elif isinstance(idx, np.ndarray):
        if idx.dtype == np.bool_:
            # A boolean mask, nothing to do
            return idx
        elif idx.ndim == 1:
            return np.delete(idx, np.where(idx == None))  # noqa
        else:
            raise ValueError("Cannot identify OOB accesses when using "
                             "multidimensional index arrays")
    else:
        return idx

import os
from string import Template
from ...loopy_utils.loopy_utils import get_device_list, kernel_call, populate
from ...kernel_utils import kernel_gen as k_gen


def __get_template(fname):
    with open(fname, 'r') as file:
        return Template(file.read())


def get_import_source():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return __get_template(os.path.join(script_dir, 'test_import.py.in'))


def get_read_ics_source():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return __get_template(os.path.join(script_dir, 'read_ic_setup.py.in'))


def clean_dir(dirname, remove_dir=True):
    if not os.path.exists(dirname):
        return
    for file in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, file)):
            os.remove(os.path.join(dirname, file))
    if remove_dir:
        os.rmdir(dirname)


class kernel_runner(object):
    """
    Simple wrapper that runs one of our kernels to find values (e.g. kf_fall,
    or X_sri)

    Parameters
    ----------
    func : Callable
        The function to use that generates the :class:`knl_info` to run
    args : dict of :class:`numpy.ndarray`
        The arguements to pass to the kernel
    kwargs : dict
        Any other arguments to pass to the func

    Returns
    -------
    vals : list of :class:`numpy.ndarray`
        The values computed by the kernel
    """

    def __init__(self, func, test_size, args={}, kwargs={}):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.test_size = test_size

    def __call__(self, eqs, loopy_opts, namestore, test_size):
        device = get_device_list()[0]

        infos = self.func(eqs, loopy_opts, namestore, test_size=test_size,
                          **self.kwargs)

        try:
            iter(infos)
        except TypeError:
            infos = [infos]

        # create a dummy generator
        gen = k_gen.make_kernel_generator(
            name='dummy',
            loopy_opts=loopy_opts,
            kernels=infos,
            test_size=self.test_size
        )
        gen._make_kernels()
        kc = kernel_call('dummy',
                         [None],
                         **self.args)
        kc.set_state(gen.array_split, loopy_opts.order)
        self.out_arg_names = [[
            x for x in k.get_written_variables()
            if x not in k.temporary_variables]
            for k in gen.kernels]
        return populate(gen.kernels, kc, device=device)[0]


def parse_split_index(arr, ind, order):
    """
    A helper method to get the index of an element in a split array for all initial
    conditions

    Parameters
    ----------
    arr: :class:`numpy.ndarray`
        The split array to use
    ind: int
        The element index
    order: ['C', 'F']
        The numpy data order

    Returns
    -------
    index: tuple of int / slice
        A proper indexing for the split array
    """

    # the index is a linear combination of the first and last indicies
    # in the split array
    if order == 'F':
        # For 'F' order, where vw is the vector width:
        # (0, 1, ... vw - 1) in the first index corresponds to the
        # last index = 0
        # (vw, vw+1, vw + 2, ... 2vw - 1) corresponds to the last index = 1,
        # etc.
        return (ind % arr.shape[0], slice(None), ind // arr.shape[0])
    else:
        # For 'C' order, where (s, l) corresponds to the second and last
        # index in the array:
        #
        # ((0, 0), (1, 0), (2, 0)), etc. corresponds to index (0, 1, 2)
        # for IC 0
        # ((0, 1), (1, 1), (2, 1)), etc. corresponds to index (0, 1, 2)
        # for IC 1, etc.

        return (slice(None), ind, slice(None))


class get_comparable(object):
    """
    A wrapper for the kernel_call's _get_comparable function that fixes
    comparison for split arrays

    Properties
    ----------
    compare_mask: list of :class:`numpy.ndarray`
        The default comparison mask
    ref_answer: :class:`numpy.ndarray`
        The answer to compare to, used to determine the proper shape
    """

    def __init__(self, compare_mask, ref_answer):
        self.compare_mask = compare_mask
        if not isinstance(self.compare_mask, list):
            self.compare_mask = [self.compare_mask]

        self.ref_answer = ref_answer
        if not isinstance(self.ref_answer, list):
            self.ref_answer = [ref_answer]

    def __call__(self, kc, outv, index):
        mask = self.compare_mask[index]
        ans = self.ref_answer[index]

        # check for vectorized data order
        if outv.ndim == ans.ndim:
            # return the default, as it can handle it
            return kernel_call('', [], compare_mask=[mask])._get_comparable(outv, 0)

        ind_list = []
        # get comparable index
        for ind in mask:
            ind_list.append(parse_split_index(outv, ind, kc.current_order))

        ind_list = zip(*ind_list)
        # filter slice arrays from parser
        for i in range(len(ind_list)):
            if all(x == slice(None) for x in ind_list[i]):
                ind_list[i] = slice(None)

        return outv[ind_list]

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

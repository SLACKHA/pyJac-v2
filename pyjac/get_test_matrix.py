def get_test_matrix(home, work_dir):
    """Runs a set of mechanisms and an ordered dictionary for
    performance and functional testing

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    Returns
    -------
    mechanisms : dict
        A dictionary indicating which mechanism are available for testing,
        The structure is as follows:
            mech_name : {'mech' : file path
                         'chemkin' : chemkin file path
                         'ns' : number of species in the mechanism
                         'thermo' : the thermo file if avaialable}
    params  : OrderedDict
        The parameters to put in an oploop
    max_vec_width : int
        The maximum vector width to test

    """
    work_dir = os.path.abspath(work_dir)

    #find the mechanisms to test
    mechanism_list = {}
    if not os.path.exists(work_dir):
        print ('Error: work directory {} for '.format(work_dir) +
               'performance testing not found, exiting...')
        sys.exit(-1)
    for name in os.listdir(work_dir):
        if os.path.isdir(os.path.join(work_dir, name)):
            #check for cti
            files = [f for f in os.listdir(os.path.join(work_dir, name)) if
                        os.path.isfile(os.path.join(work_dir, name, f))]
            for f in files:
                if f.endswith('.cti'):
                    mechanism_list[name] = {}
                    mechanism_list[name]['mech'] = f
                    mechanism_list[name]['chemkin'] = f.replace('.cti', '.dat')
                    gas = ct.Solution(os.path.join(work_dir, name, f))
                    mechanism_list[name]['ns'] = gas.n_species

                    thermo = next((tf for tf in files if 'therm' in tf), None)
                    if thermo is not None:
                        mechanism_list[name]['thermo'] = thermo

    vec_widths = [4, 8, 16]
    num_cores = []
    nc = 1
    while nc < multiprocessing.cpu_count() / 2:
        num_cores.append(nc)
        nc *= 2
    platforms = ['intel']
    rate_spec = ['fixed', 'hybrid']#, 'full']

    ocl_params = [('lang', 'opencl'),
                  ('vecsize', vec_widths),
                  ('order', ['F', 'C']),
                  ('wide', [True, False]),
                  ('platform', platforms),
                  ('rate_spec', rate_spec),
                  ('split_kernels', [True, False]),
                  ('num_cores', num_cores)
                  ]
    return mechanism_list, OrderedDict(ocl_params), vec_widths[-1]
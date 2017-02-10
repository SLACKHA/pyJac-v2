import os
def get_paths():
    path = os.path.join('/etc', 'OpenCL', 'vendors')
    vendors = [f for f in os.listdir(path) if os.path.isfile(
                    os.path.join(path, f))]

    def __get_vendor_name(vendor):
        if 'intel' in vendor.lower():
            return 'intel'
        elif 'amd' in vendor.lower():
            return 'amd'
        elif 'nvidia' in vendor.lower():
            return 'nvidia'

    paths = {}
    #now scan vendors, and get lib paths
    for v in vendors:
        with open(os.path.join(path, v), 'r') as file:
            vendor = file.read()
        p = os.path.dirname(os.path.realpath(vendor))
        if p != os.getcwd():
            #found a real path
            paths[__get_vendor_name(v)] = p
    return paths

CL_INC_DIR = ['/opt/opencl-headers/']
CL_LIBNAME = ['OpenCL']
CL_VERSION = '1.2'
CL_FLAGS = []
CC_FLAGS = []
CL_PATHS = get_paths()
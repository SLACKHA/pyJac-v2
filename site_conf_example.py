CL_INC_DIR = ['/opt/opencl-headers/']
CL_LIBNAME = ['OpenCL']
CL_VERSION = '1.2'
CL_FLAGS = []
CC_FLAGS = []
# relies on OCL-ICD
CL_LIB_DIR = ['/usr/local/lib']
# Adept is used in functional testing / unit testing for verification of Jacobian
# entries versus autodifferentiated results -- note that a standard install
# (i.e. to /usr/local/lib) probably won't need these specified
ADEPT_INC_DIR = ['/path/to/adept/include']
ADEPT_LIB_DIR = ['/path/to/adept/lib']
ADEPT_LIBNAME = ['adept']

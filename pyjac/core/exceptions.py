class MissingPlatformError(Exception):
    """
    The pyopencl platform requested for testing could not be found
    """

    def __init__(self, platform):
        self.platform = platform
        self.message = 'PyOpenCL platform {} requested for testing not found'.format(
            self.platform)
        super(MissingPlatformError, self).__init__(self.message)


class MissingDeviceError(Exception):
    """
    No devices found on the specified pyopencl platform
    """

    def __init__(self, device_type, platform):
        self.device_type = device_type
        self.platform = platform
        self.message = 'Cannot find devices of type {} on platform {}'.format(
                        self.device_type, self.platform)
        super(MissingDeviceError, self).__init__(self.message)


class CompilationError(Exception):
    """
    Error during compilation
    """

    def __init__(self, files):
        if isinstance(files, str):
            files = [files]
        self.message = 'Error compiling file(s): {}.'.format(','.join(files))
        super(CompilationError, self).__init__(self.message)


class BrokenPlatformError(Exception):
    """
    The combination of platform and vectorization options is broken
    """

    def __init__(self, loopy_opts):
        platform = loopy_opts.platform
        options = 'wide = {}, deep = {}'.format(loopy_opts.width is not None,
                                                loopy_opts.depth is not None)
        self.message = ('The platform {} is currently broken for'
                        ' vectorization options {}'.format(platform, options))
        super(BrokenPlatformError, self).__init__(self.message)

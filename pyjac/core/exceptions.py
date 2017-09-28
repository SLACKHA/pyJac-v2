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

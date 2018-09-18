"""
Contains custom errors / exceptions / error processing
"""


import six


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


class LinkingError(Exception):
    """
    Error during linking
    """

    def __init__(self, files):
        if isinstance(files, str):
            files = [files]
        self.message = 'Error linking file(s): {}.'.format(','.join(files))
        super(LinkingError, self).__init__(self.message)


class LibraryGenerationError(Exception):
    """
    Error during library generation
    """

    def __init__(self):
        self.message = 'Error generating pyJac library.'
        super(LibraryGenerationError, self).__init__(self.message)


class BrokenPlatformError(Exception):
    """
    The combination of platform and vectorization options is broken
    """

    def __init__(self, loopy_opts):
        platform = loopy_opts.platform
        options = 'wide = {}, deep = {}, explicit simd = {}'.format(
            bool(loopy_opts.width),
            bool(loopy_opts.depth),
            bool(loopy_opts.is_simd))
        self.message = ('The platform {} is currently broken for'
                        ' vectorization options {}'.format(platform, options))
        super(BrokenPlatformError, self).__init__(self.message)


def validation_error_to_string(error):
    """
    Responsible for converting a :class:`cerberus.ValidatonError` to something human
    readable

    Returns
    -------
    error: str
        The stringified error
    """

    def __stringify(root):
        error_list = []
        for k, v in six.iteritems(root):
            if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                error_list.extend(__stringify(x) for x in v)
            elif isinstance(v, dict):
                error_list.extend(__stringify(v))
            else:
                error_list.append('{}: {}'.format(k, ' | '.join(
                    vi for vi in v)))
        return error_list

    message = 'Error validating document:\n'
    message += '\n'.join(['\t{}'.format(e) for e in __stringify(error)])

    return message


class ValidationError(Exception):
    """
    Raised if a user-specified platform or test matrix fails to validate against
    our internal schemas
    """

    def __init__(self, file, schemaname):
        self.message = (
            'File {} failed to validate against schema {}, see '
            'debug output for more info.'.format(file, schemaname))
        super(ValidationError, self).__init__(self.message)


class UnknownOverrideException(Exception):
    def __init__(self, otype, path):
        self.message = (
            'Override type "{}" for path {} is unknown'.format(otype, path))
        super(UnknownOverrideException, self).__init__(self.message)


class InvalidOverrideException(Exception):
    def __init__(self, otype, value, allowed):
        from pyjac.utils import stringify_args
        self.message = (
            'Value "{}" for override type "{}" is not allowed. '
            'Allowed values are: {}'.format(otype, value, stringify_args(
                allowed)))
        super(InvalidOverrideException, self).__init__(self.message)


class OverrideCollisionException(Exception):
    def __init__(self, override_type, path):
        self.message = ('Conflicting/duplicate overrides of "{}" specified. '
                        'Dectected for "{}"'.format(
                            override_type, path))
        super(OverrideCollisionException, self).__init__(self.message)


class DuplicateTestException(Exception):
    def __init__(self, rtype, etype, filename):
        self.message = ('Multiple test types of "{}"" for evaluation type "{}" '
                        'detected in test matrix file {}'.format(
                            rtype, etype, filename))
        super(DuplicateTestException, self).__init__(self.message)


class InvalidTestEnvironmentException(Exception):
    def __init__(self, ttype, key, file, envvar):
        self.message = ('Test type "{}"" has overrides for key "{}"" specified in'
                        'test matrix file "{}", however this override cannot be '
                        'applied, as it would invalidate the test environment '
                        'key "{}"'.format(ttype, key, file, envvar))
        super(InvalidTestEnvironmentException, self).__init__(self.message)


class InvalidInputSpecificationException(Exception):
    def __init__(self, bad_inputs):
        from pyjac.utils import stringify_args, listify
        self.message = ('Inputs: ({}) were incorrectly, or conflictingly specified. '
                        'See debug output for more information'.format(
                            stringify_args(listify(bad_inputs))))
        super(InvalidInputSpecificationException, self).__init__(self.message)

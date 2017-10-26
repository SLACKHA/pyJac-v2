"""
Setuptools helper to deal with a default / user specified siteconf for OpenCL
execution.

Borrowed heavily from `pyopencl`_'s aksetup_helper.py, from which the siteconf
idea is based upon.

. _pyopencl https://documen.tician.de/pyopencl/


"""

import sys


# {{{ configure guts

def default_or(a, b):
    if a is None:
        return b
    else:
        return a


def expand_str(s, options):
    import re

    def my_repl(match):
        sym = match.group(1)
        try:
            repl = options[sym]
        except KeyError:
            from os import environ
            repl = environ[sym]

        return expand_str(repl, options)

    return re.subn(r"\$\{([a-zA-Z0-9_]+)\}", my_repl, s)[0]


def expand_value(v, options):
    if isinstance(v, str):
        return expand_str(v, options)
    elif isinstance(v, list):
        result = []
        for i in v:
            try:
                exp_i = expand_value(i, options)
            except:
                pass
            else:
                result.append(exp_i)

        return result
    else:
        return v


def expand_options(options):
    return dict(
            (k, expand_value(v, options)) for k, v in options.items())


class ConfigSchema:
    def __init__(self, options, conf_file="siteconf.py", conf_dir="."):
        self.optdict = dict((opt.name, opt) for opt in options)
        self.options = options
        self.conf_dir = conf_dir
        self.conf_file = conf_file

        from os.path import expanduser
        self.user_conf_file = expanduser("~/.aksetup-defaults.py")

        if not sys.platform.lower().startswith("win"):
            self.global_conf_file = "/etc/aksetup-defaults.py"
        else:
            self.global_conf_file = None

    def get_conf_file(self):
        import os
        return os.path.join(self.conf_dir, self.conf_file)

    def set_conf_dir(self, conf_dir):
        self.conf_dir = conf_dir

    def get_default_config(self):
        return dict((opt.name, opt.default) for opt in self.options)

    def read_config_from_pyfile(self, filename):
        result = {}
        filevars = {}
        infile = open(filename, "r")
        try:
            contents = infile.read()
        finally:
            infile.close()

        exec(compile(contents, filename, "exec"), filevars)

        for key, value in filevars.items():
            if key in self.optdict:
                result[key] = value

        return result

    def update_conf_file(self, filename, config):
        result = {}
        filevars = {}

        try:
            exec(compile(open(filename, "r").read(), filename, "exec"), filevars)
        except IOError:
            pass

        if "__builtins__" in filevars:
            del filevars["__builtins__"]

        for key, value in config.items():
            if value is not None:
                filevars[key] = value

        keys = filevars.keys()
        keys.sort()

        outf = open(filename, "w")
        for key in keys:
            outf.write("%s = %s\n" % (key, repr(filevars[key])))
        outf.close()

        return result

    def update_user_config(self, config):
        self.update_conf_file(self.user_conf_file, config)

    def update_global_config(self, config):
        if self.global_conf_file is not None:
            self.update_conf_file(self.global_conf_file, config)

    def get_default_config_with_files(self):
        result = self.get_default_config()

        import os

        confignames = []
        if self.global_conf_file is not None:
            confignames.append(self.global_conf_file)
        confignames.append(self.user_conf_file)

        for fn in confignames:
            if os.access(fn, os.R_OK):
                result.update(self.read_config_from_pyfile(fn))

        return result

    def have_global_config(self):
        import os
        result = os.access(self.user_conf_file, os.R_OK)

        if self.global_conf_file is not None:
            result = result or os.access(self.global_conf_file, os.R_OK)

        return result

    def have_config(self):
        import os
        return os.access(self.get_conf_file(), os.R_OK)

    def update_from_python_snippet(self, config, py_snippet, filename):
        filevars = {}
        exec(compile(py_snippet, filename, "exec"), filevars)

        for key, value in filevars.items():
            if key in self.optdict:
                config[key] = value
            elif key == "__builtins__":
                pass
            else:
                raise KeyError("invalid config key in %s: %s" % (
                        filename, key))

    def update_config_from_and_modify_command_line(self, config, argv):
        cfg_prefix = "--conf:"

        i = 0
        while i < len(argv):
            arg = argv[i]

            if arg.startswith(cfg_prefix):
                del argv[i]
                self.update_from_python_snippet(
                        config, arg[len(cfg_prefix):], "<command line>")
            else:
                i += 1

        return config

    def read_config(self):
        import os
        cfile = self.get_conf_file()

        result = self.get_default_config_with_files()
        if os.access(cfile, os.R_OK):
            with open(cfile, "r") as inf:
                py_snippet = inf.read()
            self.update_from_python_snippet(result, py_snippet, cfile)

        return result

    def add_to_configparser(self, parser, def_config=None):
        if def_config is None:
            def_config = self.get_default_config_with_files()

        for opt in self.options:
            default = default_or(def_config.get(opt.name), opt.default)
            opt.add_to_configparser(parser, default)

    def get_from_configparser(self, options):
        result = {}
        for opt in self.options:
            result[opt.name] = opt.take_from_configparser(options)
        return result

    def write_config(self, config):
        outf = open(self.get_conf_file(), "w")
        for opt in self.options:
            value = config[opt.name]
            if value is not None:
                outf.write("%s = %s\n" % (opt.name, repr(config[opt.name])))
        outf.close()

    def make_substitutions(self, config):
        return dict((opt.name, opt.value_to_str(config[opt.name]))
                    for opt in self.options)


class Option(object):
    def __init__(self, name, default=None, help=None):
        self.name = name
        self.default = default
        self.help = help

    def as_option(self):
        return self.name.lower().replace("_", "-")

    def metavar(self):
        last_underscore = self.name.rfind("_")
        return self.name[last_underscore+1:]

    def get_help(self, default):
        result = self.help
        if self.default:
            result += " (default: %s)" % self.value_to_str(
                    default_or(default, self.default))
        return result

    def value_to_str(self, default):
        return default

    def add_to_configparser(self, parser, default=None):
        default = default_or(default, self.default)
        default_str = self.value_to_str(default)
        parser.add_option(
            "--" + self.as_option(), dest=self.name,
            default=default_str,
            metavar=self.metavar(), help=self.get_help(default))

    def take_from_configparser(self, options):
        return getattr(options, self.name)


class StringListOption(Option):
    def value_to_str(self, default):
        if default is None:
            return None

        return ",".join([str(el).replace(",", r"\,") for el in default])

    def get_help(self, default):
        return Option.get_help(self, default) + " (several ok)"

    def take_from_configparser(self, options):
        opt = getattr(options, self.name)
        if opt is None:
            return None
        else:
            if opt:
                import re
                sep = re.compile(r"(?<!\\),")
                result = sep.split(opt)
                result = [i.replace(r"\,", ",") for i in result]
                return result
            else:
                return []


class IncludeDir(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(
            self, "%s_INC_DIR" % lib_name, default,
            help=help or ("Include directories for %s" % (
                human_name or humanize(lib_name))))


class LibraryDir(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(
            self, "%s_LIB_DIR" % lib_name, default,
            help=help or ("Library directories for %s" % (
                human_name or humanize(lib_name))))


class Libraries(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(
            self, "%s_LIBNAME" % lib_name, default,
            help=help or ("Library names for %s (without lib or .so)" % (
                human_name or humanize(lib_name))))


# {{{ tools

def flatten(list):
    """For an iterable of sub-iterables, generate each member of each
    sub-iterable in turn, i.e. a flattened version of that super-iterable.
    Example: Turn [[a,b,c],[d,e,f]] into [a,b,c,d,e,f].
    """
    for sublist in list:
        for j in sublist:
            yield j


def humanize(sym_str):
    words = sym_str.lower().replace("_", " ").split(" ")
    return " ".join([word.capitalize() for word in words])

# }}}


def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, StringListOption

    default_cxxflags = ['-std=gnu++11']

    if 'darwin' in sys.platform:
        import platform
        osx_ver, _, _ = platform.mac_ver()
        osx_ver = '.'.join(osx_ver.split('.')[:2])

        sysroot_paths = [
                "/Applications/Xcode.app/Contents/Developer/Platforms/"
                "MacOSX.platform/Developer/SDKs/MacOSX%s.sdk" % osx_ver,
                "/Developer/SDKs/MacOSX%s.sdk" % osx_ver
                ]

        default_libs = []
        default_cxxflags = default_cxxflags + [
                '-stdlib=libc++', '-mmacosx-version-min=10.7',
                '-arch', 'i386', '-arch', 'x86_64'
                ]

        from os.path import isdir
        for srp in sysroot_paths:
            if isdir(srp):
                default_cxxflags.extend(['-isysroot', srp])
                break

        default_ldflags = default_cxxflags[:] + ["-Wl,-framework,OpenCL"]

    else:
        default_libs = ["OpenCL"]
        if "linux" in sys.platform:
            # Requested in
            # https://github.com/inducer/pyopencl/issues/132#issuecomment-314713573
            # to make life with Altera FPGAs less painful by default.
            default_ldflags = ["-Wl,--no-as-needed"]
        else:
            default_ldflags = []

    return ConfigSchema([
        Option("CL__VERSION", None,
               "Dotted CL version (e.g. 1.2) which you'd like to use."),

        IncludeDir("CL", []),
        LibraryDir("CL", []),
        Libraries("CL", default_libs),

        StringListOption("CXXFLAGS", default_cxxflags,
                         help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", default_ldflags,
                         help="Any extra linker options to include"),
        ])


# {{{ siteconf handling

def get_config(schema=None, warn_about_no_config=True):
    if schema is None:
        schema = get_config_schema()

    config = expand_options(schema.read_config())
    schema.update_config_from_and_modify_command_line(config, sys.argv)
    return config

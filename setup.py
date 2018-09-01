"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from setup_helper import get_config, ConfigSchema, get_config_schema

with open('pyjac/_version.py') as version_file:
    exec(version_file.read())

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# get user's siteconf.py from CMD/file, and write to pyjac's siteconf.py
schema = get_config_schema()
conf = get_config(schema, warn_about_no_config=False)
schema.set_conf_dir(path.join(here, 'pyjac'))
schema.write_config(conf)

setup(
    name='pyJac',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description=('Create analytical Jacobian matrix source code for chemical '
                 'kinetics'),
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/SLACKHA/pyJac',

    # Author details
    author='Nick Curtis, Kyle E. Niemeyer',
    author_email='nicholas.curtis@uconn.edu, kyle.niemeyer@gmail.com',

    # Choose your license
    license='MIT License',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        'Natural Language :: English',
        'Operating System :: OS Independent',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # What does your project relate to?
    keywords='chemical_kinetics analytical_Jacobian',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['docs']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=['peppercorn'],
    install_requires=[
        'numpy',
        'loo.py>=2018.1',
        'six',
        'pyyaml',
        'cgen',
        'cerberus',
        'enum34;python_version<"3.4"'],

    tests_require=[
          'nose',
          'nose-exclude',
          'nose-testconfig',
          'Cython',
          'parameterized',
          'optionloop >= 1.0.7',
          'cantera >= 2.3.0',
          'scipy',
          'tables',
          'psutil',
          'pyopencl'],

    # use nose for tests
    test_suite='nose.collector',

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'pyjac': ['*.yaml'],
        'pyjac.pywrap': ['*.in'],
        'pyjac.functional_tester': ['*.yaml'],
        'pyjac.kernel_utils.c': ['*.c', '*.h', '*.in'],
        'pyjac.kernel_utils.common': ['*.c', '*.h', '*.in'],
        'pyjac.kernel_utils.opencl': ['*.ocl', '*.oclh', '*.in'],
        'pyjac.loopy_utils': ['*.in'],
        'pyjac.tests': ['*.cti', '*.inp'],
        'pyjac.tests.test_utils': ['*.in', '*.pyx'],
        'pyjac.examples': ['*.yaml'],
        'pyjac.schemas': ['*.yaml']
    },
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'pyjac=pyjac.__main__:main',
        ],
    },
)

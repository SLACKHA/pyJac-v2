"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from codecs import open
from os import path
from setup_helper import get_config, ConfigSchema, get_config_schema

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'pyjac', '_version.py')) as version_file:
    exec(version_file.read())

# Get the long description from the relevant files
with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

with open(path.join(here, 'CITATION.md')) as citation_file:
    citation = citation_file.read()

desc = readme + '\n\n' + changelog + '\n\n' + citation
try:
    import pypandoc
    long_description = pypandoc.convert_text(desc, 'rst', format='md')
    with open(path.join(here, 'README.rst'), 'w') as rst_readme:
        rst_readme.write(long_description)
except (ImportError, OSError, IOError):
    long_description = desc

# get user's siteconf.py from CMD/file, and write to pyjac's siteconf.py
schema = get_config_schema()
conf = get_config(schema, warn_about_no_config=False)
schema.set_conf_dir(path.join(here, 'pyjac'))
schema.write_config(conf)

setup(
    name='pyJac',
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
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
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
        'cogapp',
        'cerberus>1.1',
        'Cython',
        'enum34;python_version<"3.4"'],

    tests_require=[
          'nose',
          'nose-exclude',
          'nose-testconfig',
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
    zip_safe=False,

    entry_points={
        'console_scripts': [
            'pyjac=pyjac.__main__:main',
        ],
    },
)

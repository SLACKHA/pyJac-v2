import os
from string import Template

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
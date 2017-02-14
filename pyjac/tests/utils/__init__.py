import os
from string import Template

def get_import_source()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'test_import.py.in'), 'r') as file:
        return Template(file.read())
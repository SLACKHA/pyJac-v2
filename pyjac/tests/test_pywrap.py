# Python 2 compatibility
from __future__ import print_function
from __future__ import division

import sys

from pyjac.pywrap import pywrap_gen  # noqa


class TestPywrap_gen(object):
    """
    """
    def test_imported(self):
        """Ensure pywrap_gen module imported.
        """
        assert 'pyjac.pywrap.pywrap_gen' in sys.modules

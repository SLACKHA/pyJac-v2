import sys

from . import utils


def main(args=None):
    if args is None:
        utils.setup_logging()
        utils.create()


if __name__ == '__main__':
    sys.exit(main())

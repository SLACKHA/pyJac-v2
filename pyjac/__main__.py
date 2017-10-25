import sys

from . import utils
from .core.create_jacobian import create_jacobian


def main(args=None):
    if args is None:
        utils.setup_logging()
        args = utils.get_parser()
        create_jacobian(lang=args.lang,
                        mech_name=args.input,
                        therm_name=args.thermo,
                        vector_size=args.vector_size,
                        wide=args.wide,
                        deep=args.deep,
                        unr=args.unroll,
                        build_path=args.build_path,
                        last_spec=args.last_species,
                        platform=args.platform,
                        data_order=args.data_order,
                        rate_specialization=args.rate_specialization,
                        split_rate_kernels=args.split_rate_kernels,
                        split_rop_net_kernels=args.split_rop_net_kernels,
                        conp=args.conp,
                        use_atomics=args.use_atomics,
                        jac_type=args.jac_type,
                        jac_format=args.jac_format,
                        skip_jac=True
                        )


if __name__ == '__main__':
    sys.exit(main())

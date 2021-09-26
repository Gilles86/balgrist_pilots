import argparse

def make_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    return parser


def run_main(main, parser=None):
    if parser is None:
        parser = make_default_parser()

    args = parser.parse_args()
    main(args.subject, args.session, args.bids_folder)

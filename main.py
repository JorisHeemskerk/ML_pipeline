import argparse
import timestamp # import to measure starting runtime

from job_factory import JobFactory


def main(args)-> None:
    # Construct and run parser
    job_factory = JobFactory(args.config_file_path)
    job_factory.parse()
    job_factory.run()


if __name__ == "__main__":
    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument(
        '-c',
        '--config', 
        dest='config_file_path', 
        type=str, 
        default="config/config.yaml", 
        help="Path to config file. (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args)
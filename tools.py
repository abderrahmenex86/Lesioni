import argparse

from src.helpers import (
    download_dataset,
    generate_analytics_plots,
    verify_dataset_structure,
)


def main():
    argument_parser = argparse.ArgumentParser(description="Lesioni Utility Processing CLI")
    argument_parser.add_argument(
        "--mode", type=str, required=True, choices=["download", "verify", "plot"], help="Utility processing option"
    )
    argument_parser.add_argument(
        "--root_dir", type=str, default="dataset", help="Target parent folder root for datasets"
    )

    parsed_arguments = argument_parser.parse_args()

    if parsed_arguments.mode == "download":
        download_dataset(destination_directory=parsed_arguments.root_dir)

    elif parsed_arguments.mode == "verify":
        verify_dataset_structure(dataset_directory=f"{parsed_arguments.root_dir}/plantseg")

    elif parsed_arguments.mode == "plot":
        generate_analytics_plots()


if __name__ == "__main__":
    main()

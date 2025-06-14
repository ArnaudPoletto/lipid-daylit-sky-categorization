import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.texture_descriptor import (
    get_sky_finder_texture_descriptors,
    get_fitted_umap_reducer,
    get_kmeans_groups,
    plot_sky_finder_texture_descriptors,
)

def parse_args() -> None:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Plot texture descriptor space for sky finder."
    )
    parser.add_argument(
        "--group-by",
        "-g",
        type=str,
        choices=["sky_type", "cluster"],
        default="sky_type",
        help="Group by sky type or cluster.",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=3,
        help="Number of nearest neighbors to consider when grouping by cluster.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode.",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()
    group_by = args.group_by
    k = args.k
    interactive = args.interactive

    # Get sky finder texture descriptors and fitted UMAP reducer
    sky_finder_texture_descriptors, sky_types, image_paths = get_sky_finder_texture_descriptors()
    fitted_umap_reducer = get_fitted_umap_reducer(
        sky_finder_texture_descriptors=sky_finder_texture_descriptors
    )

    if group_by == "sky_type":
        colors = [
            "blue" if sky_type == "clear" else
            "red" if sky_type == "overcast" else "orange"
            for sky_type in sky_types
        ]
        color_labels = {
            "blue": "clear",
            "orange": "partial",
            "red": "overcast"
        }
    elif group_by == "cluster":
        cluster_labels = get_kmeans_groups(texture_descriptors=sky_finder_texture_descriptors, k=k)
        colors = [
            f"C{label}" if label >= 0 else "black" for label in cluster_labels
        ]
        color_labels = {f"C{label}": f"Cluster {label}" for label in set(cluster_labels) if label >= 0}

    plot_sky_finder_texture_descriptors(
        fitted_umap=fitted_umap_reducer,
        sky_finder_texture_descriptors=sky_finder_texture_descriptors,
        colors=colors,
        color_labels=color_labels,
        image_paths=image_paths if interactive else None,
    )

if __name__ == "__main__":
    main()
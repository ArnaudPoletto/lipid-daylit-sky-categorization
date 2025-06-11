import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.texture_descriptor import (
    get_sky_finder_texture_descriptors,
    get_fitted_umap_reducer,
    plot_sky_finder_texture_descriptors,
)

def main() -> None:
    # Get sky finder texture descriptors and fitted UMAP reducer
    sky_finder_texture_descriptors, sky_types = get_sky_finder_texture_descriptors()
    colors = [
        "blue" if sky_type == "clear" else
        "red" if sky_type == "overcast" else "orange"
        for sky_type in sky_types
    ]
    fitted_umap_reducer = get_fitted_umap_reducer(
        sky_finder_texture_descriptors=sky_finder_texture_descriptors
    )

    plot_sky_finder_texture_descriptors(
        fitted_umap=fitted_umap_reducer,
        sky_finder_texture_descriptors=sky_finder_texture_descriptors,
        colors=colors,
    )

if __name__ == "__main__":
    main()
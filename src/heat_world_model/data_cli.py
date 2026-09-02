import argparse
from pathlib import Path

from .dataset import (
    C45DatasetConfig,
    DatasetConfig,
    generate_c45_dataset,
    generate_dataset,
    save_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate complete temperature-field trajectories."
    )
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--nx", type=int, default=41)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=("constant", "c45-radiation"),
        default="constant",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectories = args.trajectories
    if args.model == "c45-radiation" and trajectories == 100:
        trajectories = 120
    config_type = C45DatasetConfig if args.model == "c45-radiation" else DatasetConfig
    config = config_type(
        trajectories=trajectories,
        steps=args.steps,
        nx=args.nx,
        dt_s=args.dt,
        seed=args.seed,
    )
    dataset = (
        generate_c45_dataset(config)
        if isinstance(config, C45DatasetConfig)
        else generate_dataset(config)
    )
    output = args.output or Path(
        "outputs/c45_radiative_dataset.npz"
        if args.model == "c45-radiation"
        else "outputs/world_model_dataset.npz"
    )
    save_dataset(output, config, dataset)
    counts = {
        name: int((dataset["split"] == label).sum())
        for label, name in enumerate(("train", "validation", "test", "ood_test"))
    }
    print(
        f"saved={output} states_shape={dataset['states_c'].shape} "
        f"split_counts={counts}"
    )


if __name__ == "__main__":
    main()

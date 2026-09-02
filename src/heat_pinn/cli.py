import argparse
from pathlib import Path

from .plotting import plot_loss_history, plot_temperature_fields
from .problem import HeatEquation1D
from .train import TrainingConfig, evaluate_model, save_run_data, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the 1D transient heat PINN baseline.")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-domain", type=int, default=2000)
    parser.add_argument("--n-boundary", type=int, default=200)
    parser.add_argument("--n-initial", type=int, default=200)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        n_domain=args.n_domain,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        seed=args.seed,
        log_every=args.log_every,
    )
    problem = HeatEquation1D()
    model, history, training_summary = train_model(problem, config)
    evaluation = evaluate_model(model, problem)
    metrics = evaluation["metrics"]
    assert isinstance(metrics, dict)
    save_run_data(
        args.output_dir, problem, config, history, training_summary, metrics
    )
    plot_temperature_fields(evaluation, args.output_dir / "temperature_field.png")
    plot_loss_history(history, args.output_dir / "loss_curve.png")
    print(
        f"relative_l2={metrics['relative_l2']:.3e} "
        f"temperature_rmse_c={metrics['temperature_rmse_c']:.3f} "
        f"elapsed_seconds={training_summary.elapsed_seconds:.2f}"
    )


if __name__ == "__main__":
    main()

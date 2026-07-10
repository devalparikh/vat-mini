"""Command-line entry points for the full local learning loop."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from vat_mini.checkpoint import load_checkpoint
from vat_mini.config import ExperimentConfig, load_config
from vat_mini.data import ACTION_NAMES, build_dataloaders, save_dataset
from vat_mini.device import seed_everything, select_device
from vat_mini.evaluation import evaluate_demonstrations, evaluate_rollouts
from vat_mini.model import VisionActionTransformer
from vat_mini.trainer import Trainer


def _load_experiment(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(args.config, args.set or [])


def inspect_environment(_: argparse.Namespace) -> None:
    details = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(select_device("auto")),
        "actions": list(ACTION_NAMES),
    }
    print(json.dumps(details, indent=2))


def generate_data(args: argparse.Namespace) -> None:
    config = _load_experiment(args)
    destination = args.output or config.data.dataset_path or "data/gridworld.npz"
    path = save_dataset(destination, config.data, config.seed)
    print(f"saved dataset to {path}")


def train(args: argparse.Namespace) -> dict[str, float]:
    config = _load_experiment(args)
    seed_everything(config.seed)
    device = select_device(config.device)
    train_loader, validation_loader = build_dataloaders(config.data, config.seed)
    trainer = Trainer(config, VisionActionTransformer(config.model), device)
    checkpoint = args.checkpoint or config.training.initial_checkpoint
    if checkpoint:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"initial checkpoint does not exist: {checkpoint}")
        trainer.initialize_from_checkpoint(checkpoint)
    print(f"training stage={config.training.stage} device={device}")
    return trainer.fit(train_loader, validation_loader)


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    config = _load_experiment(args)
    seed_everything(config.seed)
    device = select_device(config.device)
    _, validation_loader = build_dataloaders(config.data, config.seed)
    model = VisionActionTransformer(config.model)
    checkpoint = args.checkpoint or str(Path(config.output_dir) / "latest.pt")
    payload = load_checkpoint(checkpoint, model)
    model.to(device)
    metrics = evaluate_demonstrations(model, validation_loader, device)
    metrics.update(
        evaluate_rollouts(
            model,
            device,
            config.data.grid_size,
            config.data.image_size,
            episodes=args.episodes,
            seed=config.seed + 2,
        )
    )
    print(json.dumps({"checkpoint_stage": payload.get("stage"), **metrics}, indent=2))
    return metrics


def smoke(args: argparse.Namespace) -> None:
    metrics = train(args)
    required = {"validation_loss", "validation_token_accuracy", "rollout_success_rate"}
    if not required.issubset(metrics) or not all(torch.isfinite(torch.tensor(metrics[key])) for key in required):
        raise RuntimeError("smoke run produced invalid metrics")
    if metrics["validation_token_accuracy"] <= metrics["validation_majority_class_baseline"]:
        raise RuntimeError("policy did not learn beyond the validation majority-class baseline")
    if metrics["rollout_success_rate"] <= 0.10:
        raise RuntimeError("policy did not learn meaningful closed-loop control")
    print("smoke test completed: train, checkpoint, and closed-loop evaluation are healthy")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="repeatable dotted override"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vat-mini")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="show local accelerator support")
    inspect_parser.set_defaults(handler=inspect_environment)

    generate_parser = commands.add_parser("generate-data", help="write deterministic demonstrations")
    _add_config_arguments(generate_parser)
    generate_parser.add_argument("--output")
    generate_parser.set_defaults(handler=generate_data)

    train_parser = commands.add_parser("train", help="run behavior cloning or post-training")
    _add_config_arguments(train_parser)
    train_parser.add_argument("--checkpoint", help="initialize model weights from a checkpoint")
    train_parser.set_defaults(handler=train)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a saved policy")
    _add_config_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint")
    evaluate_parser.add_argument("--episodes", type=int, default=32)
    evaluate_parser.set_defaults(handler=evaluate)

    smoke_parser = commands.add_parser("smoke", help="run a tiny end-to-end train/evaluate cycle")
    _add_config_arguments(smoke_parser)
    smoke_parser.add_argument("--checkpoint")
    smoke_parser.set_defaults(handler=smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

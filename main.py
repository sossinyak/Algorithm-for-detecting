import argparse
import random

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Зафиксировать seed для воспроизводимых запусков."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str = "config.yaml") -> dict:
    """Загрузить YAML-конфигурацию."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск экспериментов проекта")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к файлу конфигурации",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="comparison",
        choices=["comparison", "parameter", "ablation", "monte_carlo"],
        help="Тип эксперимента",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Необязательный путь к датасету",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Необязательное ограничение числа пар изображений",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Не показывать графики результатов",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.data_path:
        config["data"]["data_path"] = args.data_path

    if args.max_samples is not None:
        config.setdefault("experiments", {})["max_samples"] = args.max_samples

    if args.no_plot:
        config.setdefault("experiments", {})["no_plot"] = True

    set_seed(config["seed"])

    print("=" * 60)
    print("ПРОЕКТ ОБНАРУЖЕНИЯ ИЗМЕНЕНИЙ")
    print("=" * 60)

    if args.experiment == "comparison":
        from experiments.run_comparison import run_comparison_experiment

        run_comparison_experiment(config)
    elif args.experiment == "parameter":
        from analysis.parameter_study import run_parameter_study

        run_parameter_study(config)
    elif args.experiment == "ablation":
        from analysis.ablation_study import run_ablation_study

        run_ablation_study(config)
    elif args.experiment == "monte_carlo":
        from analysis.monte_carlo_optimization import run_monte_carlo_optimization

        run_monte_carlo_optimization(config)

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ablation_study import run_ablation_study
from analysis.parameter_study import run_parameter_study
from analysis.statistical_tests import run_statistical_tests
from experiments.run_comparison import run_comparison_experiment
from scripts.generate_visual_report import build_visual_report


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_external_status(external_data_path: str | None) -> None:
    out = ROOT / "results" / "external_dataset_status.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not external_data_path:
        out.write_text(
            "Сравнение на внешнем датасете не запускалось: external_data_path не задан.\n"
            "Для запуска подготовьте датасет со структурой LEVIR: test/A, test/B, test/label, "
            "затем передайте --external_data_path <path>.\n",
            encoding="utf-8",
        )
        return

    path = Path(external_data_path)
    required = [path / "test" / "A", path / "test" / "B", path / "test" / "label"]
    if not all(p.exists() for p in required):
        out.write_text(
            f"Сравнение на внешнем датасете не запускалось: {path} не содержит test/A, test/B, test/label.\n"
            "Ожидается структура каталогов формата LEVIR.\n",
            encoding="utf-8",
        )
        return

    out.write_text(
        f"Внешний датасет доступен по пути {path}. Запустите сравнение командой:\n"
        f"python main.py --experiment comparison --data_path \"{path}\"\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Запустить все эксперименты воспроизводимости для классического алгоритма.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Ограничение числа пар для сравнения.")
    parser.add_argument("--study_samples", type=int, default=120, help="Ограничение числа пар для ablation и исследования параметров.")
    parser.add_argument("--visual_samples", type=int, default=5)
    parser.add_argument("--skip_comparison", action="store_true")
    parser.add_argument("--external_data_path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("experiments", {})["no_plot"] = True

    if args.max_samples is not None:
        config["experiments"]["max_samples"] = args.max_samples

    if not args.skip_comparison:
        print("\n=== Эксперимент сравнения ===")
        run_comparison_experiment(config)

    study_config = load_config(args.config)
    study_config.setdefault("experiments", {})["max_samples"] = args.study_samples
    study_config["experiments"]["no_plot"] = True

    print("\n=== Ablation study: вклад улучшений ===")
    run_ablation_study(study_config)

    print("\n=== Статистические тесты ===")
    run_statistical_tests()

    print("\n=== Исследование параметров ===")
    run_parameter_study(study_config)

    print("\n=== Визуальный отчет ===")
    build_visual_report(
        data_path=config["data"]["data_path"],
        split="test",
        max_samples=args.visual_samples,
    )

    write_external_status(args.external_data_path)
    print("\nВсе запрошенные артефакты экспериментов сохранены в results/.")


if __name__ == "__main__":
    main()

"""Запускает эксперименты и сохраняет параметры, результаты и график."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev

from genetic import DIMENSION, LOWER, UPPER, Config, Run, genetic_algorithm, random_search


def save_csv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def plot_convergence(path: Path, results: dict[str, dict[int, Run]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "ga_p010": "ГА, p мутации = 0,10",
        "ga_p030": "ГА, p мутации = 0,30",
        "random": "Случайный поиск",
    }
    figure, axes = plt.subplots(figsize=(9, 5), layout="constrained")
    for name, runs in results.items():
        values_by_generation = list(zip(*(run.history for run in runs.values())))
        averages = [mean(values) for values in values_by_generation]
        # Нормальная аппроксимация для среднего по независимым запускам.
        margins = [1.96 * stdev(values) / sqrt(len(values)) for values in values_by_generation]
        generations = range(len(averages))
        (line,) = axes.plot(generations, averages, label=labels[name], linewidth=2)
        axes.fill_between(
            generations,
            [average - margin for average, margin in zip(averages, margins)],
            [average + margin for average, margin in zip(averages, margins)],
            color=line.get_color(),
            alpha=0.18,
        )
    axes.set(
        title="Сходимость на функции Стыблинского–Танга",
        xlabel="Поколение (для случайного поиска — порция того же размера)",
        ylabel="Лучшее значение f к данному поколению",
    )
    axes.legend(title="Среднее и приближённый 95%-й ДИ")
    axes.grid(alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026, help="первое значение seed")
    parser.add_argument("--runs", type=int, default=30, help="число запусков на метод")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.runs < 20:
        parser.error("требуется не менее 20 запусков на метод")

    base = Config()
    configurations = {"ga_p010": base, "ga_p030": replace(base, mutation_probability=0.30)}
    seeds = range(args.seed, args.seed + args.runs)
    results = {
        name: {seed: genetic_algorithm(config, seed) for seed in seeds}
        for name, config in configurations.items()
    }
    results["random"] = {seed: random_search(base, seed) for seed in seeds}

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    batch_size = base.population_size - base.elite_count
    parameters = {
        "variant": 14,
        "dimension": DIMENSION,
        "bounds": [LOWER, UPPER],
        "seeds": list(seeds),
        "evaluation_budget_per_run": base.evaluation_budget,
        "ga": {name: asdict(config) for name, config in configurations.items()},
        "random": {
            "initial_samples": base.population_size,
            "batch_size": batch_size,
            "batches": base.generations,
        },
    }
    (output / "config.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    save_csv(
        output / "runs.csv",
        ("method", "seed", "evaluations", "best_f", *(f"x{i + 1}" for i in range(DIMENSION))),
        (
            (name, seed, run.evaluations, run.best.value, *run.best.coordinates)
            for name, runs in results.items()
            for seed, run in runs.items()
        ),
    )
    save_csv(
        output / "history.csv",
        ("method", "seed", "generation", "evaluations", "best_f"),
        (
            (name, seed, generation, base.population_size + generation * batch_size, value)
            for name, runs in results.items()
            for seed, run in runs.items()
            for generation, value in enumerate(run.history)
        ),
    )
    summary = []
    for name, runs in results.items():
        values = [run.best.value for run in runs.values()]
        summary.append(
            (name, len(values), min(values), mean(values), median(values), stdev(values), max(values))
        )
    save_csv(
        output / "summary.csv",
        ("method", "runs", "best", "mean", "median", "std", "worst"),
        summary,
    )
    plot_convergence(output / "convergence.png", results)
    print(f"{len(seeds)} запусков на метод, {base.evaluation_budget} вычислений на запуск.")
    for name, _, best, average, _, _, _ in summary:
        print(f"{name}: best = {best:.6f}, mean = {average:.6f}")


if __name__ == "__main__":
    main()

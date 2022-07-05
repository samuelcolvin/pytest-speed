from dataclasses import dataclass
from statistics import mean, stdev
from time import perf_counter_ns
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from .save import save_benchmarks
from .utils import GitSummary, benchmark_change, calc_div_units, group_benchmarks, render_time

__args__ = 'BenchmarkConfig', 'Benchmark', 'BenchmarkRun'


@dataclass
class BenchmarkConfig:
    """
    Store configuration info for benchmarking
    """

    warmup_time_ns = 1_000_000_000
    warmup_max_iterations = 5_000
    max_rounds = 10_000
    max_time_ns = 3_000_000_000
    outlier_percentage = 10

    ideal_rounds = 100
    min_rounds = 10
    ideal_iterations = 10_000
    min_iterations = 200


@dataclass
class Benchmark:
    """
    Store results of a single benchmark.
    """

    name: str
    group: Optional[str]
    best_ns: int
    worse_ns: int
    mean_ns: float
    stddev_ns: float
    bench_time_ns: int
    rounds: int
    iter_per_round: int
    outlier_rounds: int
    outlier_prop: float
    warnings: Sequence[str] = ()

    def summary(self) -> str:
        best_ns = self.best_ns / self.iter_per_round
        units, div = calc_div_units(best_ns)
        parts = dict(
            group=self.group,
            name=self.name,
            best=render_time(best_ns, units, div),
            stdev=render_time(self.stddev_ns / self.iter_per_round, units, div),
            iterations=f'{self.rounds * self.iter_per_round:,}',
            warnings=', '.join(self.warnings),
        )
        return ' '.join(f'[blue]{k}[/blue]=[green]{v}[/green]' for k, v in parts.items() if v)


class BenchmarkCollection:
    """
    Manage a benchmark run and store data about it.
    """

    def __init__(self, config: BenchmarkConfig, save: bool):
        self.config = config
        self.save = save
        self.benchmarks: list[Benchmark] = []
        # these are updated later by _prepare_units
        self.units = 's'
        self.div = 1_000_000_000
        self.table = Table()
        self.group_best: Optional[float] = None
        self.git = GitSummary.build()

    def run_benchmark(self, name: str, group: Optional[str], func: Callable[..., Any], *args: Any) -> Benchmark:
        """
        Run a single benchmark and record data about it.
        """
        warnings: List[str] = []
        iter_per_round, rounds = self._warmup(func, args)

        times = []
        loop_range = range(iter_per_round)
        start_time = perf_counter_ns()
        toc = start_time

        for _ in range(rounds):
            tic = perf_counter_ns()
            for _ in loop_range:
                func(*args)
            toc = perf_counter_ns()
            times.append(toc - tic)
            if toc - start_time > self.config.max_time_ns * 2:
                warnings.append('Benchmark timed out')
                break

        bench_time_ns = toc - start_time
        best_ns = min(times)
        outlier_threshold = int(best_ns * (1 + self.config.outlier_percentage / 100))
        outlier_rounds = sum(1 for t in times if t > outlier_threshold)

        outlier_prop = outlier_rounds / rounds
        if outlier_prop > 0.1:
            warnings.append(f'{outlier_prop:0.0%} high outliers')

        benchmark = Benchmark(
            name=name,
            group=group,
            best_ns=best_ns,
            worse_ns=max(times),
            mean_ns=mean(times),
            stddev_ns=stdev(times),
            bench_time_ns=bench_time_ns,
            rounds=len(times),
            iter_per_round=iter_per_round,
            outlier_rounds=outlier_rounds,
            outlier_prop=outlier_prop,
            warnings=warnings,
        )
        self.benchmarks.append(benchmark)
        return benchmark

    def _warmup(self, func: Callable[..., Any], args: Sequence[Any]) -> Tuple[int, int]:
        """
        Run warmup iterations and return tuple of (iter_per_round, rounds).
        """
        times = []
        start_time = perf_counter_ns()
        for _ in range(self.config.warmup_max_iterations):
            tic = perf_counter_ns()
            func(*args)
            toc = perf_counter_ns()
            times.append(toc - tic)
            if toc - start_time > self.config.warmup_time_ns:
                break

        mean_warmup = mean(times)
        del times
        # we want to run ideal_rounds rounds of iterations, each group consisting of up to ideal_iterations iterations
        # we want them to finish in less than max_time_ns
        # that means each round should take max_time_ns / ideal_rounds

        round_time = self.config.max_time_ns / self.config.ideal_rounds
        iter_per_round = min(self.config.ideal_iterations, int(round_time / mean_warmup))

        rounds = self.config.ideal_rounds
        if iter_per_round < self.config.min_iterations:
            rounds = self.config.min_rounds
            round_time = self.config.max_time_ns / rounds
            iter_per_round = max(self.config.min_iterations, int(round_time / mean_warmup))
        return iter_per_round, rounds

    def finish(self) -> None:
        if not self.benchmarks:
            print('No benchmarks run')
            return

        console = Console()
        title = ['Benchmarks', str(self.git)]
        if self.save:
            bm_id, save_path = save_benchmarks(self.benchmarks, self.config, self.git)
            title.append(f'Save ID: [blue]{bm_id:03d}[/blue]')
            console.print(f'[italic][dim]Saved benchmarks to [/dim][cyan]{escape(save_path)}[/cyan][dim].[/dim]')

        self.table = Table(title=' '.join(t for t in title if t), padding=(0, 2), expand=True, border_style='cyan')

        show_groups = any(bm.group for bm in self.benchmarks)
        self._prepare_units()

        if show_groups:
            self.table.add_column('Group', style='bold')
        self.table.add_column('Test Name')
        self.table.add_column(f'Best ({self.units}/iter)', justify='right')
        if show_groups:
            self.table.add_column('Relative', justify='right')
        self.table.add_column(f'Stddev ({self.units}/iter)', justify='right')
        self.table.add_column('Iterations', justify='right')
        self.table.add_column('Note')

        if show_groups:
            for bm_group in group_benchmarks(self.benchmarks).values():
                group_len = len(bm_group)
                bm_group.sort(key=lambda bm: bm.best_ns / bm.iter_per_round)
                for index, bm in enumerate(bm_group):
                    self._add_group_row(index == 0, index + 1 == group_len, bm)

            self.benchmarks.sort(key=lambda bm: (bm.group, bm.best_ns / bm.iter_per_round))
        else:
            for bm in self.benchmarks:
                self._add_no_group_row(bm)

        console.print(self.table)

    def _add_group_row(self, first_in_group: bool, last_in_group: bool, benchmark: Benchmark) -> None:
        best_ns = benchmark.best_ns / benchmark.iter_per_round
        if first_in_group:
            # new group
            self.group_best = best_ns
            group_col = benchmark.group
            rel = ''
            # if just one item in the group, no style
            row_style = 'normal' if last_in_group else 'green'
        else:
            # show the worse result in red
            row_style = 'red' if last_in_group else 'cyan'
            group_col = ''
            rel = benchmark_change(cast(float, self.group_best), best_ns)

        self.table.add_row(
            group_col,
            Text(benchmark.name or '(no name)', style=row_style),
            Text(self._render_time(best_ns), style=row_style),
            Text(rel, style=row_style),
            Text(self._render_time(benchmark.stddev_ns / benchmark.iter_per_round), style=row_style),
            Text(f'{benchmark.rounds * benchmark.iter_per_round:,}', style=row_style),
            self._row_note(benchmark),
            end_section=last_in_group,
        )

    def _add_no_group_row(self, benchmark: Benchmark) -> None:
        self.table.add_row(
            benchmark.name or '(no name)',
            self._render_time(benchmark.best_ns / benchmark.iter_per_round),
            self._render_time(benchmark.stddev_ns / benchmark.iter_per_round),
            f'{benchmark.rounds * benchmark.iter_per_round:,}',
            self._row_note(benchmark),
        )

    def _prepare_units(self) -> None:
        min_time = min(bm.best_ns / bm.iter_per_round for bm in self.benchmarks)
        self.units, self.div = calc_div_units(min_time)

    def _render_time(self, ns: float) -> str:
        return render_time(ns, '', self.div)

    @staticmethod
    def _row_note(benchmark: Benchmark) -> Union[str, Text]:
        if benchmark.warnings:
            return Text('\n'.join(benchmark.warnings), style='red')
        else:
            return ''

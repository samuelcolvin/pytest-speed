from dataclasses import dataclass
from datetime import datetime
from statistics import mean, stdev
from time import perf_counter_ns
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from .save import BenchmarkSummary, save_benchmarks
from .utils import GitSummary, benchmark_change, calc_div_units, format_ts, group_benchmarks, render_time

__args__ = 'BenchmarkConfig', 'Benchmark', 'BenchmarkRun', 'BenchmarkTable', 'compare_benchmarks'


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
        if self.save:
            bm_id, save_path = save_benchmarks(self.benchmarks, self.config, self.git)
            console.print(f'[italic][dim]Saved benchmarks to [/dim][cyan]{escape(save_path)}[/cyan][dim].[/dim]')
        else:
            bm_id = None

        BenchmarkTable(console, self.git, self.benchmarks, bm_id).print()


class BenchmarkTable:
    """
    Logic for printing a table summarising benchmarks.
    """

    def __init__(self, console: Console, git: GitSummary, benchmarks: List[Benchmark], bm_id: Optional[int] = None):
        self.console = console
        title = ['Benchmarks', str(git)]
        if bm_id is not None:
            title.append(f'Save ID: [blue]{bm_id:03d}[/blue]')

        self.table = Table(
            title=' '.join(t for t in title if t), padding=(0, 2), expand=True, min_width=120, border_style='cyan'
        )
        self.benchmarks = benchmarks
        min_time = min(bm.best_ns / bm.iter_per_round for bm in benchmarks)
        self.units, self.div = calc_div_units(min_time)
        self.group_best: Optional[float] = None

    def print(self) -> None:
        show_groups = any(bm.group for bm in self.benchmarks)

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

        self.console.print(self.table)

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

    def _render_time(self, ns: float) -> str:
        return render_time(ns, '', self.div)

    @staticmethod
    def _row_note(benchmark: Benchmark) -> Union[str, Text]:
        if benchmark.warnings:
            return Text('\n'.join(benchmark.warnings), style='red')
        else:
            return ''


def compare_benchmarks(before: BenchmarkSummary, after: BenchmarkSummary) -> None:
    """
    Compare two sets of benchmarks.
    """
    now = datetime.now()
    console = Console()
    table = Table(title='Benchmarks being compared', title_justify='left', padding=(0, 2), border_style='cyan')
    table.add_column('', style='bold')
    table.add_column('Before')
    table.add_column('After')
    table.add_row('ID', f'{before.id:03d}', f'{after.id:03d}')
    table.add_row('Branch', before.git.branch, after.git.branch)
    table.add_row('Commit SHA', before.git.commit[:7], after.git.commit[:7])
    table.add_row('Commit Message', before.git.commit_message, after.git.commit_message)
    table.add_row('Benchmark Timestamp', format_ts(before.timestamp, now), format_ts(after.timestamp, now))

    console.print('')
    console.print(table)

    min_time = min(
        [bm.best_ns / bm.iter_per_round for bm in before.benchmarks]
        + [bm.best_ns / bm.iter_per_round for bm in after.benchmarks]
    )
    units, div = calc_div_units(min_time)

    table = Table(title='Benchmarks Comparison', title_justify='left', padding=(0, 2), border_style='cyan')
    table.add_column('Group', style='bold')
    table.add_column('Benchmark')
    table.add_column(f'Before ({units}/iter)', justify='right')
    table.add_column(f'After ({units}/iter)', justify='right')
    table.add_column('Change', justify='right')

    test_keys = set()
    after_lookup = {benchmark_key(bm): bm for bm in after.benchmarks}
    before_not_after = 0

    for bm_group in group_benchmarks(before.benchmarks).values():
        for index, bm in enumerate(bm_group):
            key = benchmark_key(bm)
            after_bm = after_lookup.get(key)
            test_keys.add(key)
            before_ns = bm.best_ns / bm.iter_per_round

            group_name = (bm.group or '') if index == 0 else ''
            end_section = index == len(bm_group) - 1
            if after_bm:
                after_ns = after_bm.best_ns / after_bm.iter_per_round
                style = None
                if after_ns > before_ns * 1.1:
                    style = 'red'
                elif after_ns < before_ns * 0.9:
                    style = 'green'
                table.add_row(
                    group_name,
                    bm.name,
                    render_time(before_ns, '', div),
                    render_time(after_ns, '', div),
                    benchmark_change(before_ns, after_ns),
                    style=style,
                    end_section=end_section,
                )
            else:
                before_not_after += 1

    console.print('')
    console.print(table)
    if before_not_after:
        console.print(f'{before_not_after} benchmarks in before but not after.', style='red')
    after_not_before = sum(benchmark_key(bm) not in test_keys for bm in after.benchmarks)
    if after_not_before:
        console.print(f'{after_not_before} benchmarks in after but not before.', style='red')


def benchmark_key(bm: 'Benchmark') -> str:
    if bm.group:
        return f'{bm.group}:{bm.name}'
    else:
        return bm.name

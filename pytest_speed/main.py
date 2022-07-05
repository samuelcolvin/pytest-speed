import subprocess
from dataclasses import dataclass
from statistics import mean, stdev
from time import perf_counter_ns
from typing import Optional, Any, Callable, Sequence, Tuple, List, Union

from rich.console import Console
from rich.table import Table
from rich.text import Text


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
    high_percentage = 10

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
    high_rounds: int
    warnings: Sequence[str] = ()

    def summary(self):
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


class BenchmarkRun:
    """
    Manage a benchmark run and store data about it.
    """
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmarks: list[Benchmark] = []
        # these are updated later by _prepare_units
        self.units = 's'
        self.div = 1_000_000_000
        self.table = Table()
        self.current_group = None
        self.group_best = None

    def run_benchmark(self, name: str, group: Optional[str], func: Callable[[...], Any], *args: Any) -> Benchmark:
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
        high_threshold = int(best_ns * (1 + self.config.high_percentage / 100))
        high_rounds = sum(1 for t in times if t > high_threshold)

        high_prop = high_rounds / rounds
        if high_prop > 0.1:
            warnings.append(f'{high_prop:0.2%} of iterations are high')

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
            high_rounds=high_rounds,
            warnings=warnings,
        )
        self.benchmarks.append(benchmark)
        return benchmark

    def _warmup(self, func: Callable[[...], Any], args: Sequence[Any]) -> Tuple[int, int]:
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

    def print_results(self):
        if not self.benchmarks:
            print('No benchmarks run')
            return

        branch, commit = git_summary()
        self.table = Table(title=f'Benchmarks {branch} ({commit})', padding=(0, 2), expand=True, border_style='cyan')

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

        self.benchmarks.sort(key=lambda bm: (bm.group, bm.mean_ns))

        for (index, benchmark) in enumerate(self.benchmarks):
            if show_groups:
                self._add_group_row(index, benchmark)
            else:
                self._add_no_group_row(benchmark)

        console = Console()
        console.print(self.table)

    def _add_group_row(self, index: int, benchmark: Benchmark):
        group_in_last = index + 1 >= len(self.benchmarks) or self.benchmarks[index + 1].group != benchmark.group
        best_ns = benchmark.best_ns / benchmark.iter_per_round
        if benchmark.group != self.current_group:
            # new group
            self.current_group = benchmark.group
            self.group_best = best_ns
            group_col = self.current_group
            rel = ''
            # if just one item in the group, no style
            row_style = 'normal' if group_in_last else 'green'
        else:
            # show the worse result in red
            row_style = 'red' if group_in_last else 'cyan'
            group_col = ''
            if best_ns > self.group_best * 2:
                rel = f'x{best_ns / self.group_best:0.2f}'
            else:
                rel = f'+{(best_ns - self.group_best) / self.group_best:0.2%}'

        self.table.add_row(
            group_col,
            Text(benchmark.name or '(no name)', style=row_style),
            Text(self._render_time(best_ns), style=row_style),
            Text(rel, style=row_style),
            Text(self._render_time(benchmark.stddev_ns / benchmark.iter_per_round), style=row_style),
            Text(f'{benchmark.rounds * benchmark.iter_per_round:,}', style=row_style),
            self._row_note(benchmark),
            end_section=group_in_last,
        )

    def _add_no_group_row(self, benchmark: Benchmark):
        self.table.add_row(
            benchmark.name or '(no name)',
            self._render_time(benchmark.best_ns / benchmark.iter_per_round),
            self._render_time(benchmark.stddev_ns / benchmark.iter_per_round),
            f'{benchmark.rounds * benchmark.iter_per_round:,}',
            self._row_note(benchmark),
        )

    def _prepare_units(self):
        min_time = min(bm.best_ns / bm.iter_per_round for bm in self.benchmarks)
        self.units, self.div = calc_div_units(min_time)

    def _render_time(self, ns) -> str:
        return render_time(ns, '', self.div)

    @staticmethod
    def _row_note(benchmark: Benchmark) -> Union[str, Text]:
        if benchmark.warnings:
            return Text('\n'.join(benchmark.warnings), style='red')
        else:
            return ''


def git_summary() -> Tuple[str, str]:
    p = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], check=True, stdout=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return '', ''
    branch = p.stdout.strip()
    p = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, stdout=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return branch, ''
    commit = p.stdout.strip()
    return branch, commit[:7]


def render_time(time_ns: float, units: str, div: int) -> str:
    value = time_ns / div
    if value < 1:
        dp = 3
    else:
        dp = 2 if value < 100 else 1
    return f'{value:.{dp}f}{units}'


def calc_div_units(time_ns: float) -> Tuple[str, int]:
    if time_ns < 1_000:
        return 'ns', 1
    elif time_ns < 1_000_000:
        return 'µs', 1_000
    elif time_ns < 1_000_000_000:
        return 'ms', 1_000_000
    else:
        return 's', 1_000_000_000

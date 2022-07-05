from datetime import datetime

import click
from rich.console import Console
from rich.table import Table

from .benchmark import BenchmarkTable, compare_benchmarks
from .save import BenchmarkSummary, load_all_benchmarks, load_benchmark
from .utils import format_ts
from .version import VERSION


@click.group()
@click.version_option(VERSION)
def cli() -> None:
    """
    CLI for pytest-speed, can be used to list saved benchmarks and compare two benchmarks.
    """
    pass


@cli.command(name='list')
def list_() -> None:
    """
    List all saved benchmarks.
    """
    benchmark_summaries = load_all_benchmarks()
    console = Console()
    table = Table(title='Saved Benchmarks', padding=(0, 2), border_style='cyan')

    table.add_column('ID', style='bold')
    table.add_column('Timestamp')
    table.add_column('Branch')
    table.add_column('Commit SHA')
    table.add_column('Commit Message')
    table.add_column('Benchmarks', justify='right')

    now = datetime.now()
    benchmark_summaries.sort(key=lambda bs_: bs_.id)
    for bs in benchmark_summaries:
        table.add_row(
            f'{bs.id:03d}',
            format_ts(bs.timestamp, now),
            bs.git.branch,
            f'{bs.git.commit[:7]}{" [dirty]" if bs.git.dirty else ""}',
            bs.git.commit_message,
            f'{len(bs.benchmarks):,}',
        )
    console.print(table)


@cli.command()
@click.argument('benchmark_id', type=int)
def display(benchmark_id: int) -> None:
    """
    Display a table summarising a single benchmark run.

    Same table as is printed after a run.
    """
    bms = get_benchmark(benchmark_id)
    BenchmarkTable(Console(), bms.git, bms.benchmarks).print()


@cli.command()
@click.argument('id_before', type=int)
@click.argument('id_after', type=int)
def compare(id_before: int, id_after: int) -> None:
    """
    Load two benchmarks and compare them.

    IDs should match those from the "ID" column of `pytest-speed list`.
    """
    before = get_benchmark(id_before)
    after = get_benchmark(id_after)

    compare_benchmarks(before, after)


def get_benchmark(benchmark_id: int) -> BenchmarkSummary:
    try:
        return load_benchmark(benchmark_id)
    except FileNotFoundError:
        raise click.UsageError(f'No benchmark with ID {benchmark_id}')

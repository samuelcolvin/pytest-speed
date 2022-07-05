from datetime import datetime
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.table import Table

from .save import load_benchmarks
from .utils import benchmark_change, calc_div_units, format_ts, group_benchmarks, render_time
from .version import VERSION

if TYPE_CHECKING:
    from .benchmark import Benchmark


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
    benchmark_summaries = load_benchmarks()
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


@cli.command(name='compare')
@click.argument('id_before', type=int)
@click.argument('id_after', type=int)
def compare(id_before: int, id_after: int) -> None:
    """
    Load two benchmarks and compare them.

    IDs should match those from the "ID" column of `pytest-speed list`.
    """
    benchmark_summaries = load_benchmarks()
    try:
        bm_before = next(bm for bm in benchmark_summaries if bm.id == id_before)
    except StopIteration:
        raise click.UsageError(f'No benchmark with ID {id_before}')
    try:
        bm_after = next(bm for bm in benchmark_summaries if bm.id == id_after)
    except StopIteration:
        raise click.UsageError(f'No benchmark with ID {id_after}')

    now = datetime.now()
    console = Console()
    table = Table(title='Benchmarks being compared', title_justify='left', padding=(0, 2), border_style='cyan')
    table.add_column('', style='bold')
    table.add_column('Before')
    table.add_column('After')
    table.add_row('ID', f'{bm_before.id:03d}', f'{bm_after.id:03d}')
    table.add_row('Branch', bm_before.git.branch, bm_after.git.branch)
    table.add_row('Commit SHA', bm_before.git.commit[:7], bm_after.git.commit[:7])
    table.add_row('Commit Message', bm_before.git.commit_message, bm_after.git.commit_message)
    table.add_row('Benchmark Timestamp', format_ts(bm_before.timestamp, now), format_ts(bm_after.timestamp, now))

    console.print('')
    console.print(table)

    min_time = min(
        [bm.best_ns / bm.iter_per_round for bm in bm_before.benchmarks]
        + [bm.best_ns / bm.iter_per_round for bm in bm_after.benchmarks]
    )
    units, div = calc_div_units(min_time)

    table = Table(title='Benchmarks Comparison', title_justify='left', padding=(0, 2), border_style='cyan')
    table.add_column('Group', style='bold')
    table.add_column('Benchmark')
    table.add_column(f'Before ({units}/iter)', justify='right')
    table.add_column(f'After ({units}/iter)', justify='right')
    table.add_column('Change', justify='right')

    tests = set()
    after_lookup = {benchmark_key(bm): bm for bm in bm_after.benchmarks}

    for bm_group in group_benchmarks(bm_before.benchmarks).values():
        for index, bm in enumerate(bm_group):
            key = benchmark_key(bm)
            after = after_lookup.get(key)
            tests.add(key)
            before_ns = bm.best_ns / bm.iter_per_round

            group_name = (bm.group or '') if index == 0 else ''
            end_section = index == len(bm_group) - 1
            if after:
                after_ns = after.best_ns / after.iter_per_round
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
                table.add_row(group_name, bm.name, render_time(before_ns, '', div), '-', '-', style='dim')

    for bm_group in group_benchmarks(bm_after.benchmarks).values():
        for index, bm in enumerate(bm_group):
            key = benchmark_key(bm)
            if key not in tests:
                table.add_row(
                    bm.group or '', bm.name, '-', render_time(bm.best_ns / bm.iter_per_round, '', div), '-', style='dim'
                )

    console.print('')
    console.print(table)


def benchmark_key(bm: 'Benchmark') -> str:
    if bm.group:
        return f'{bm.group}:{bm.name}'
    else:
        return bm.name

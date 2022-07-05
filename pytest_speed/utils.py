import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .benchmark import Benchmark


@dataclass
class GitSummary:
    found: bool
    branch: str = ''
    commit: str = ''
    commit_message: str = ''
    dirty: bool = False

    def __str__(self) -> str:
        if self.found:
            s = f'{self.branch} ({self.commit[:7]})'
            if self.dirty:
                s += ' [dirty]'
            return s
        else:
            return ''

    @classmethod
    def build(cls) -> 'GitSummary':
        if not Path('.git').exists():
            return GitSummary(found=False)
        p = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], check=True, stdout=subprocess.PIPE, text=True)
        branch = p.stdout.strip()
        p = subprocess.run(
            ['git', 'describe', '--dirty', '--always', '--long', '--abbrev=40'],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        dirty = '-dirty' in p.stdout
        if dirty:
            commit = p.stdout.strip().split('-', 1)[0]
        else:
            commit = p.stdout.strip()
        p = subprocess.run(
            ['git', 'log', '--format=%B', '-n', '1', commit], check=True, stdout=subprocess.PIPE, text=True
        )
        commit_message = p.stdout.strip()
        return cls(True, branch, commit, commit_message, dirty)


def render_time(time_ns: float, units: str, div: int) -> str:
    value = time_ns / div
    if value < 1:
        dp = 3
    else:
        dp = 2 if value < 100 else 1
    return f'{value:.{dp}f}{units}'


def benchmark_change(before: float, after: float) -> str:
    if after > before * 2:
        return f'x{after / before:0.2f}'
    else:
        return f'{(after - before) / before:+0.2%}'


def group_benchmarks(benchmarks: 'List[Benchmark]') -> 'Dict[Optional[str], List[Benchmark]]':
    groups: 'Dict[Optional[str], List[Benchmark]]' = {}
    for bm in benchmarks:
        group = groups.get(bm.group)
        if group:
            group.append(bm)
        else:
            groups[bm.group] = [bm]
    return groups


def calc_div_units(time_ns: float) -> Tuple[str, int]:
    if time_ns < 1_000:
        return 'ns', 1
    elif time_ns < 1_000_000:
        return 'µs', 1_000
    elif time_ns < 1_000_000_000:
        return 'ms', 1_000_000
    else:
        return 's', 1_000_000_000


def format_ts(ts: datetime, now: datetime) -> str:
    if ts.date() == now.date():
        diff = now - ts
        if diff.seconds < 60:
            ago = f'{diff.seconds} seconds'
        else:
            mins = round(diff.seconds / 60)
            if diff.seconds < 3600:
                ago = f'{mins:.0f} mins'
            else:
                ago = f'{mins / 60:.0f} hours, {mins % 60:.0f} mins'
        return f'{ts:%H:%M} ({ago} ago)'
    else:
        return f'{ts:%Y-%m-%d %H:%M}'

import re
from typing import Optional

import pytest
import rich

from .main import BenchmarkConfig, BenchmarkRun


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'benchmark: pytest-speed marker to define benchmark groups (compatible with pytest-benchmark)'
    )
    config.addinivalue_line('markers', 'speed: pytest-speed marker to define benchmark groups')


stub_help = 'pytest-speed stub for pytest-benchmark, ignored'


def pytest_addoption(parser):
    parser.addoption('--benchmark-columns', action='store', default='-', help=stub_help)
    parser.addoption('--benchmark-group-by', action='store', default='-', help=stub_help)
    parser.addoption('--benchmark-warmup', action='store', default='-', help=stub_help)
    parser.addoption('--benchmark-disable', action='store_true', default='-', help=stub_help)
    parser.addoption(
        '--benchmark-enable',
        dest='bench',
        action='store_true',
        default=False,
        help='alias for "--bench", compatible with pytest-benchmark - enable benchmarks',
    )
    parser.addoption('--bench', action='store_true', default=False, help='enable benchmarks')


benchmarks: Optional[BenchmarkRun] = None


@pytest.fixture(scope='session')
def benchmark_run(request):
    if request.config.getoption('bench'):
        global benchmarks
        benchmarks = BenchmarkRun(BenchmarkConfig())
        return benchmarks


@pytest.fixture
def bench(request, capsys, benchmark_run: Optional[BenchmarkRun]):
    verbose_level = request.config.getoption('verbose')
    call_index = 0

    def run_benchmark(func, *args, name: str = None, group: str = None):
        nonlocal call_index
        if benchmark_run is None:
            func(*args)
            return

        test_name = re.sub('^test_', '', request.node.name)
        if name is not None:
            name = name.format(test=test_name, index=call_index)
        elif call_index == 0:
            name = test_name
        else:
            name = f'{test_name}_{call_index}'

        if group is None:
            group = next((m.kwargs['group'] for m in request.node.iter_markers('speed')), None)
            if group is None:
                group = next((m.kwargs['group'] for m in request.node.iter_markers('benchmark')), None)

        call_index += 1
        benchmark = benchmark_run.run_benchmark(name, group, func, *args)
        if verbose_level > 0:
            with capsys.disabled():
                rich.print(benchmark.summary(), end='')
        return benchmark

    return run_benchmark


@pytest.fixture
def benchmark(bench):
    """
    Compatibility with pytest-benchmark
    """
    return bench


def pytest_terminal_summary():
    if benchmarks:
        benchmarks.print_results()

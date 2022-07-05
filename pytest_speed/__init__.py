import re

import pytest

from .main import BenchmarkRun, BenchmarkConfig

def pytest_configure(config):
    config.addinivalue_line('markers', 'benchmark: pytest-speed marker to define benchmark groups (compatible with pytest-benchmark)')
    config.addinivalue_line('markers', 'speed: pytest-speed marker to define benchmark groups')


stub_help = 'pytest-speed stub for pytest-benchmark, ignored'


def pytest_addoption(parser):
    parser.addoption('--benchmark-columns', action='store', default='-', help=stub_help)
    parser.addoption('--benchmark-group-by', action='store', default='-', help=stub_help)
    parser.addoption('--benchmark-warmup', action='store', default='-', help=stub_help)
    parser.addoption('--benchmark-disable', action='store_true', default='-', help=stub_help)
    parser.addoption('--benchmark-enable', action='store_true', default=False, help='alias for "--bench", compatible with pytest-benchmark')
    parser.addoption('--bench', action='store_true', default=False, help='run benchmarks')


@pytest.fixture(name='_benchmark_run', scope='session')
def benchmark_run():
    return BenchmarkRun(BenchmarkConfig())


@pytest.fixture
def benchmark(request, benchmark_run: BenchmarkRun):
    call_index = 0

    def run_benchmark(func, *args, name: str = None, group: str = None):
        nonlocal call_index

        test_name = re.sub('^test_', '', request.node.name)
        if name is None:
            if call_index == 0:
                name = name
            else:
                name = f'{name}_{call_index}'
        else:
            name = name.format(test=test_name, index=call_index)

        if group is None:
            group = next((m.kwargs['group'] for m in request.node.iter_markers('speed')), None)
            if group is None:
                group = next((m.kwargs['group'] for m in request.node.iter_markers('benchmark')), None)

        call_index += 1
        benchmark_run.run_benchmark(name, group, func, *args)

    return run_benchmark


def pytest_terminal_summary(*args):
    debug(args)

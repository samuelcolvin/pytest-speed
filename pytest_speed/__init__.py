import re
from typing import Any, Callable, Optional, Protocol

import pytest
import rich

from .benchmark import Benchmark, BenchmarkCollection, BenchmarkConfig
from .version import VERSION

__version__ = VERSION


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        'markers', 'benchmark: pytest-speed marker to define benchmark groups (compatible with pytest-benchmark)'
    )
    config.addinivalue_line('markers', 'speed: pytest-speed marker to define benchmark groups')


stub_help = 'pytest-speed stub for pytest-benchmark, ignored'


def pytest_addoption(parser: Any) -> None:
    parser.addoption('--benchmark-columns', action='store', default='', help=stub_help)
    parser.addoption('--benchmark-group-by', action='store', default='', help=stub_help)
    parser.addoption('--benchmark-warmup', action='store', default='', help=stub_help)
    parser.addoption('--benchmark-disable', action='store_true', default='', help=stub_help)
    parser.addoption(
        '--benchmark-save',
        action='store',
        default='',
        help='pytest-speed stub for pytest-benchmark, value is ignored, but if set, benchmarks are saved',
    )
    parser.addoption(
        '--benchmark-enable',
        dest='bench',
        action='store_true',
        default=False,
        help='alias for "--bench", compatible with pytest-benchmark - enable benchmarks',
    )
    parser.addoption('--bench', action='store_true', default=False, help='enable benchmarks')
    parser.addoption('--bench-save', action='store_true', default=False, help='save benchmarks')


benchmarks: Optional[BenchmarkCollection] = None


class RunBench(Protocol):
    def __call__(
        self, func: Callable[..., Any], *args: Any, name: Optional[str] = None, group: Optional[str] = None
    ) -> Optional[Benchmark]:
        ...


@pytest.fixture(scope='session')
def benchmark_collection(request: Any) -> Optional[BenchmarkCollection]:
    global benchmarks

    save = any(request.config.getoption(opt) for opt in ('bench_save', 'benchmark_save'))
    if request.config.getoption('bench') or save:
        benchmarks = BenchmarkCollection(BenchmarkConfig(), save)
        return benchmarks
    else:
        return None


@pytest.fixture
def bench(request: Any, capsys: Any, benchmark_collection: Optional[BenchmarkCollection]) -> RunBench:
    verbose_level = request.config.getoption('verbose')
    call_index = 0

    def run(
        func: Callable[..., Any], *args: Any, name: Optional[str] = None, group: Optional[str] = None
    ) -> Optional[Benchmark]:
        nonlocal call_index
        if benchmark_collection is None:
            # benchmarks not enabled, just run the function and return
            func(*args)
            return None

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
        benchmark = benchmark_collection.run_benchmark(name, group, func, *args)
        if verbose_level > 0:
            with capsys.disabled():
                rich.print(benchmark.summary(), end='')
        return benchmark

    return run


@pytest.fixture
def benchmark(bench: RunBench) -> RunBench:
    """
    Compatibility with pytest-benchmark
    """
    return bench


def pytest_terminal_summary() -> None:
    if benchmarks:
        benchmarks.finish()

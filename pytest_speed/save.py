import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .utils import GitSummary

if TYPE_CHECKING:
    from .benchmark import Benchmark, BenchmarkConfig

__args__ = 'save_benchmarks', 'load_all_benchmarks', 'load_benchmark', 'BenchmarkSummary'
benchmark_save_dir = Path('.benchmarks/speed')


def save_benchmarks(benchmarks: 'List[Benchmark]', config: 'BenchmarkConfig', git: GitSummary) -> Tuple[int, str]:
    """
    save benchmarks to file.
    """
    data: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'git_info': asdict(git),
        'config': asdict(config),
        'benchmarks': [asdict(bm) for bm in benchmarks],
    }
    if benchmark_save_dir.exists():
        bm_id = sum(1 for _ in benchmark_save_dir.glob('bench*')) + 1
    else:
        bm_id = 1
        benchmark_save_dir.mkdir(parents=True)

    data['id'] = bm_id
    path = benchmark_save_dir / f'bench{bm_id:03d}.json'
    with path.open('w') as f:
        json.dump(data, f, indent=2)
    return bm_id, str(path)


@dataclass
class BenchmarkSummary:
    id: int
    timestamp: datetime
    config: 'BenchmarkConfig'
    git: GitSummary
    benchmarks: 'List[Benchmark]'


def load_all_benchmarks() -> List[BenchmarkSummary]:
    benchmark_summaries = []
    for path in benchmark_save_dir.glob('bench*'):
        m = re.search(r'bench(\d+)', path.name)
        if m:
            benchmark_id = int(m.group(1))
            benchmark_summaries.append(load_benchmark(benchmark_id))
    return benchmark_summaries


def load_benchmark(benchmark_id: int) -> BenchmarkSummary:
    from .benchmark import Benchmark, BenchmarkConfig

    path = benchmark_save_dir / f'bench{benchmark_id:03d}.json'
    with path.open() as f:
        data = json.load(f)
    return BenchmarkSummary(
        id=data['id'],
        timestamp=datetime.fromisoformat(data['timestamp']),
        config=BenchmarkConfig(**data['config']),
        git=GitSummary(**data['git_info']),
        benchmarks=[Benchmark(**bm) for bm in data['benchmarks']],
    )

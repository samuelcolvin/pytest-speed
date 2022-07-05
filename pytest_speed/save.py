import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .utils import GitSummary

if TYPE_CHECKING:
    from .benchmark import Benchmark, BenchmarkConfig

__args__ = 'save_benchmarks', 'load_all_benchmarks', 'BenchmarkSummary'
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


def load_benchmarks() -> 'List[BenchmarkSummary]':
    from .benchmark import Benchmark, BenchmarkConfig

    benchmark_summaries = []
    for path in benchmark_save_dir.glob('bench*'):
        with path.open() as f:
            data = json.load(f)
        bms = BenchmarkSummary(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            config=BenchmarkConfig(**data['config']),
            git=GitSummary(**data['git_info']),
            benchmarks=[Benchmark(**bm) for bm in data['benchmarks']],
        )
        benchmark_summaries.append(bms)
    return benchmark_summaries

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re

def _chunk_text(text, num_workers):
    lines = text.splitlines()
    chunk_size = max(1, len(lines) // num_workers)
    return ["\n".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]

def parallel_process(text, func, num_workers=None, **kwargs):
    """Generic parallel processor for text chunks."""
    nw = num_workers or multiprocessing.cpu_count()
    chunks = _chunk_text(text, nw)
    
    target_func = partial(func, **kwargs) if kwargs else func
    
    with ProcessPoolExecutor(max_workers=nw) as executor:
        results = list(executor.map(target_func, chunks))
    
    return "\n".join(results)

def dict_lookup_chunk(chunk, mapping):
    """Dictionary lookup logic for a single text chunk."""
    tokens = re.findall(r'\w+|[^\w\s]|\s+', chunk)
    result = []
    for token in tokens:
        if re.match(r'\w+', token):
            result.append(mapping.get(token.lower(), token))
        else:
            result.append(token)
    return "".join(result)

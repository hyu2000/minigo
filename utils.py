# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Miscellaneous utilities"""

from contextlib import contextmanager
import functools
import itertools
import logging
import operator
import os
import re
import sys
import time
import datetime
import numpy as np
from typing import Iterator, List, Tuple


def dbg(*objects, file=sys.stderr, flush=True, **kwargs):
    "Helper function to print to stderr and flush"
    print(*objects, file=file, flush=flush, **kwargs)


def ensure_dir_exists(directory):
    "Creates local directories if they don't exist."
    if directory.startswith('gs://'):
        return
    if not os.path.exists(directory):
        dbg("Making dir {}".format(directory))
    os.makedirs(directory, exist_ok=True)


def parse_game_result(result):
    "Parse an SGF result string into value target."
    if re.match(r'[bB]\+', result):
        return 1
    if re.match(r'[wW]\+', result):
        return -1
    return 0


def format_game_summary(all_moves: List[str], result: str, first_n: int = 12, last_n: int = 2, sgf_fname=''):
    open_moves = all_moves[: first_n]
    end_moves = all_moves[-last_n:]
    line = f'%s ..%3d .. %s \t%-6s' % (' '.join(open_moves), len(all_moves), ' '.join(end_moves), result)
    line = line.replace('pass', '--', -1)
    short_fname = os.path.basename(sgf_fname).removesuffix('.sgf')
    return f'{line}\t{short_fname}'


def product(iterable):
    "Like sum(), but with multiplication."
    return functools.reduce(operator.mul, iterable)


def _take_n(num_things, iterable):
    return list(itertools.islice(iterable, num_things))


def iter_chunks(chunk_size, iterator):
    "Yield from an iterator in chunks of chunk_size."
    iterator = iter(iterator)
    while True:
        next_chunk = _take_n(chunk_size, iterator)
        # If len(iterable) % chunk_size == 0, don't return an empty chunk.
        if next_chunk:
            yield next_chunk
        else:
            break


def grouper(n, iterable: Iterator):
    """Itertools recipe
    >>> list(grouper(3, iter('ABCDEFG')))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    >>> list(grouper(iter(range(10)), 3))  # iter() important!
    """
    return iter(lambda: list(itertools.islice(iterable, n)), [])


@contextmanager
def timer(message):
    "Context manager for timing snippets of code."
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f seconds" % (message, (tock - tick)))


@contextmanager
def logged_timer(message):
    "Context manager for timing snippets of code. Echos to logging module."
    tick = time.time()
    yield
    tock = time.time()
    logging.info("%s: %.3f seconds", message, (tock - tick))


def microseconds_since_midnight():
    now = datetime.datetime.now()
    tdelta = now - now.replace(hour=0, minute=0, second=0, microsecond=0)
    return tdelta.seconds * 1000000 + tdelta.microseconds


def soft_pick(pi: np.ndarray, temperature=1.0, softpick_topn_cutoff: int = 0) -> int:
    """ pick a move by sampling pi**temp

    pi might be modified
    """
    if softpick_topn_cutoff > 0:
        nth = np.partition(pi.flatten(), -softpick_topn_cutoff)[-softpick_topn_cutoff]
        pi[pi < nth] = 0
    if temperature != 1.0:
        pi = pi ** temperature
    cdf = pi.cumsum()

    if cdf[-2] > 1e-6:
        cdf /= cdf[-2]  # Prevents passing via softpick.
        selection = np.random.random()
        fcoord = cdf.searchsorted(selection)
        return fcoord

    assert False, f'soft_pick {pi} failed: {cdf[-2]}'


def choose_moves_with_probs(moves_with_probs: List[Tuple], temperature=1.0, softpick_topn_cutoff: int = 0, n: int = 1):
    """ same as soft_pick, but with a top-moves list, sorted by probs """
    if softpick_topn_cutoff > 0:
        moves_with_probs = moves_with_probs[:softpick_topn_cutoff]
    probs = np.array([x[1] for x in moves_with_probs])
    if temperature != 1.0:
        probs = probs ** temperature
    probs = probs / probs.sum()
    indices = np.random.choice(len(moves_with_probs), size=n, p=probs)
    moves = [moves_with_probs[i][0] for i in indices]
    return moves if n != 1 else moves[0]

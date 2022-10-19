"""
utils that use Zobrist Hash to
- count states visited: rough counting, no collision check
- filter legal moves in MCTS: exact, check for collision
"""
import logging
from typing import Dict

import numpy as np
import go
import coords
from symmetries import apply_symmetry_feat, SYMMETRIES

_hasher = go.zobrist_hasher


def board_hash_canonical(board: np.ndarray) -> np.uint64:
    """ a unique hash across 8 symmetries
    Expensive!
    """
    hashes = [_hasher.board_hash(apply_symmetry_feat(s, board)) for s in SYMMETRIES]
    return min(hashes)


def calc_legal_moves_sans_symmetry(pos: go.Position) -> np.ndarray:
    """ reduce moves that lead to the same board (under symmetry)
    Deterministic: when symmetry happens, we prefer move with higher flat index value

    slow for MCTS... maybe it's faster if not using board_hash?
    """
    lmoves = pos.all_legal_moves()
    # num_legal_moves = lmoves.sum()

    hashes = dict()  # type: Dict[np.uint64, np.ndarray]
    # lmoves[-1] == 1 is pass, ignore
    legal_move_flat_indices = np.argwhere(lmoves[:-1] == 1)[:, 0]
    # reverse the array to prefer higher indices. Purely my preference for C2/D2, etc.
    for flat_idx in legal_move_flat_indices[::-1]:
        move = coords.from_flat(flat_idx)
        pos_after_move = pos.play_move(move)
        dup = False
        new_hashes = dict()
        for s in SYMMETRIES:
            if s == 'identity':
                board = pos_after_move.board
                h = pos_after_move.zobrist_hash
            else:
                board = apply_symmetry_feat(s, pos_after_move.board)
                h = _hasher.board_hash(board)
            if h in hashes:
                dup = np.array_equal(hashes[h], board)
                if dup:
                    # print(f'move %s dup under {s}' % coords.flat_to_gtp(flat_idx))
                    break
                print('#'*60)
                print(f'ZHash rare collision: {h}')
                print(board)
                print(hashes[h])
            new_hashes[h] = board
        if dup:
            lmoves[flat_idx] = 0
        else:
            hashes.update(new_hashes)
            # print(f'move %s added' % coords.flat_to_gtp(flat_idx))
    # logging.info(f'reduce_symmetry move# {pos.n}: {num_legal_moves} -> {lmoves.sum()}')
    return lmoves


_legal_reduced_moves_cache = dict()


def legal_moves_cache_size() -> int:
    return len(_legal_reduced_moves_cache)


def legal_moves_sans_symmetry(pos: go.Position) -> np.ndarray:
    """ fast version for some open positions (even faster than all_legal_positions()
    """
    if pos.n >= 3:  # no reduction or caching beyond first 3 moves
        return pos.all_legal_moves()

    zhash = pos.zobrist_hash
    lmoves = _legal_reduced_moves_cache.get(zhash)
    if lmoves is not None:
        return lmoves

    lmoves = calc_legal_moves_sans_symmetry(pos)
    _legal_reduced_moves_cache[zhash] = lmoves
    return lmoves
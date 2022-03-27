"""
utils that use Zobrist Hash to
- count states visited: rough counting, no collision check
- filter legal moves in MCTS: exact, check for collision
"""
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


def legal_moves_sans_symmetry(pos: go.Position) -> np.ndarray:
    """ """
    lmoves = pos.all_legal_moves()

    hashes = dict()  # type: Dict[np.uint64, np.ndarray]
    # lmoves[-1] == 1 is pass, ignore
    legal_move_flat_indices = np.argwhere(lmoves[:-1] == 1)[:, 0]
    for flat_idx in legal_move_flat_indices:
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
    return lmoves

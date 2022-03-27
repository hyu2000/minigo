"""
utils that use Zobrist Hash to
- count states visited: rough counting, no collision check
- filter legal moves in MCTS: exact, check for collision
"""

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


def legal_moves_sans_symmetry(pos: 'go.Position') -> np.ndarray:
    """ """
    lmoves = pos.all_legal_moves()

    hashes = set()
    # lmoves[-1] == 1 is pass, ignore
    legal_move_flat_indices = np.argwhere(lmoves[:-1] == 1)[:, 0]
    for flat_idx in legal_move_flat_indices:
        move = coords.from_flat(flat_idx)
        pos_after_move = pos.play_move(move)
        dup = False
        new_hashes = set()
        for s in SYMMETRIES:
            if s == 'identity':
                h = pos_after_move.zobrist_hash
            else:
                h = _hasher.board_hash(apply_symmetry_feat(s, pos_after_move.board))
            if h in hashes:
                dup = True
                # print(f'move %s dup under {s}' % coords.flat_to_gtp(flat_idx))
                break
            new_hashes.add(h)
        if dup:
            lmoves[flat_idx] = 0
        else:
            hashes.update(new_hashes)
            # print(f'move %s added' % coords.flat_to_gtp(flat_idx))
    return lmoves

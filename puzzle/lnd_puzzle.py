import os
from collections import namedtuple

import numpy as np

import go


class BBox(namedtuple('BBox', ['row0', 'col0', 'row1', 'col1'])):
    """ top-left, bottom-right in minigo coord style """

    def nrows(self):
        return 1 + self.row1 - self.row0

    def ncols(self):
        return 1 + self.col1 - self.col0

    def area(self) -> int:
        return self.nrows() * self.ncols()

    def area_opposite_bbox(self) -> int:
        """ size of the bbox opposite of this one """
        return (go.N - self.nrows()) * (go.N - self.ncols())

    def grow(self, delta: int):
        return BBox(max(0, self.row0 - delta), max(0, self.col0 - delta),
                    min(go.N-1, self.row1 + delta), min(go.N-1, self.col1 + delta))


class LnDPuzzle:
    """ helper to figure out basics of an LnD puzzle """
    # tolerance to the edge of the board
    SNAP_EDGE_THRESH = 1

    @staticmethod
    def snap(k: int, to: int):
        return to if abs(k - to) <= LnDPuzzle.SNAP_EDGE_THRESH else k

    @classmethod
    def find_bound_for(cls, board: np.array, side: int) -> BBox:
        """
        """
        stone_coords = np.argwhere(board == side)
        imin, jmin = stone_coords.min(axis=0)
        imax, jmax = stone_coords.max(axis=0)
        imin, jmin = cls.snap(imin, 0), cls.snap(jmin, 0)
        imax, jmax = cls.snap(imax, go.N), cls.snap(jmax, go.N-1)
        return BBox(imin, jmin, imax, jmax)

    @staticmethod
    def enclosing_box(a: BBox, b: BBox) -> BBox:
        return BBox(min(a.row0, b.row0), min(a.col0, b.col0), max(a.row1, b.row1), max(a.col1, b.col1))

    @classmethod
    def solve_boundary(cls, board: np.array) -> tuple[BBox, BBox, int]:
        """ given a puzzle board, figure out who's surrounding, corner/center area (for the surrounded)

        Attacker is more on the peripheral, or has more stones there (maybe by a weighted sum)
        """
        black_boundary = cls.find_bound_for(board, go.BLACK)
        white_boundary = cls.find_bound_for(board, go.WHITE)
        if black_boundary == white_boundary:
            print('white_boundary == black_boundary: cannot figure out attacker/defender')
            return black_boundary, white_boundary, 0
        enclosure = cls.enclosing_box(black_boundary, white_boundary)
        attack_side = 0
        if enclosure == black_boundary:
            attack_side = go.BLACK
        elif enclosure == white_boundary:
            attack_side = go.WHITE
        if attack_side == 0:
            print('cannot figure out which side is bigger')
            return black_boundary, white_boundary, attack_side
        return black_boundary, white_boundary, attack_side

    @staticmethod
    def contested_area(board: np.array, defender_bbox: BBox, attack_side: int) -> np.array:
        """ contested area: a simple estimate, basically just exclude attacker stones inside bbox
        Could be very off if many dead attacker stones inside the bbox.

        defender_bbox is obtained from solve_boundary()
        """
        bbox = defender_bbox
        mask = np.zeros(board.shape)
        mask[bbox.row0:(1 + bbox.row1), bbox.col0:(1 + bbox.col1)] = \
            board[bbox.row0:(1 + bbox.row1), bbox.col0:(1 + bbox.col1)] != attack_side
        return mask


def rect_mask(bbox: BBox) -> np.array:
    """ all inclusive """
    mask = np.zeros((go.N, go.N), dtype=np.int8)
    mask[bbox.row0:bbox.row1+1, bbox.col0:bbox.col1+1] = 1
    return mask


def test_bbox():
    print()
    bbox = BBox(0, 3, 3, 8)
    mask0 = rect_mask(bbox)
    print(mask0)
    assert bbox.area() == 24

    oppo_area = (go.N * go.N - bbox.area()) // 2
    rows_needed = oppo_area / go.N
    print(bbox.nrows(), bbox.ncols(), 'need', rows_needed)


def test_puzzle_boundary():
    from sgf_wrapper import SGFReader

    # snap should allow edge_thresh=2?
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題5級(9).sgf'
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級.sgf'
    # sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級(4).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    black_box, white_box, attack_side = LnDPuzzle.solve_boundary(pos.board)
    assert attack_side == go.BLACK
    print("black", black_box)
    print("white", white_box)

    white_enlarged = white_box.grow(1)
    # 総合問題4級.sgf  white policy area: 0..3, 3..8
    assert white_enlarged.col0 == 3 and white_enlarged.row1 == 3

    contested_area = LnDPuzzle.contested_area(pos.board, white_box, attack_side)
    print(contested_area)
    contested_size = np.sum(contested_area)
    print(f'contested area size = {contested_size}')
    assert contested_size == 14


def test_puzzle_bulk():
    import glob
    from sgf_wrapper import SGFReader

    sgf_dir = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death'
    sgf_fnames = sorted(glob.glob(f'{sgf_dir}/総合問題5級*.sgf'))
    print('found', len(sgf_fnames))
    for sgf_fname in sgf_fnames:
        basename = os.path.split(sgf_fname)[-1]
        print(f'\nProcessing {basename}')
        reader = SGFReader.from_file_compatible(sgf_fname)
        pos = reader.first_pos()
        black_box, white_box, attack_side = LnDPuzzle.solve_boundary(pos.board)
        if attack_side == 0:
            print('no clear attack_side, skipping')
            continue
        contested_area = LnDPuzzle.contested_area(pos.board, white_box if attack_side == go.BLACK else black_box, attack_side)
        print(contested_area)

        # check defender owns the area for now
        komi = pos.komi
        score = pos.score_tromp(mask=contested_area)
        print(f'komi={komi}, attack_side={attack_side}, score={score}')
        assert komi == 0
        assert score * attack_side < 0


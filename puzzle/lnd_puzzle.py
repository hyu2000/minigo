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
    def contested_area(board: np.array, defender_bbox: BBox, attack_side: int) -> int:
        """ size of the contested area: a simple estimate, basically just exclude attacker stones inside bbox
        Could be very off if many dead attacker stones inside the bbox.

        defender_bbox is obtained from solve_boundary()
        """
        bbox = defender_bbox
        return np.sum(board[bbox.row0:(1+bbox.row1), bbox.col0:(1+bbox.col1)] != attack_side)

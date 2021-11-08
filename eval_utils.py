from typing import Tuple, Dict, Sequence, List
import attr


@attr.s
class Outcome(object):
    moves = attr.ib()
    result = attr.ib()
    count = attr.ib(default=1)


class RedundancyChecker(object):
    """ In evaluation mode, bot runs in a more-deterministic mode.
    If the first several moves are the same, game will most likely end up the same.
    Use this to avoid repetitive work.

    We can also allow to play a couple extra games, just to validate that this is indeed the case.
    """
    def __init__(self, num_open_moves, max_verify_games=4):
        self.num_open_moves = num_open_moves
        self._result_map = dict()  # type: Dict[Tuple, Outcome]
        self._num_verify_games = 0
        self._max_verify_games = max_verify_games

    @staticmethod
    def _player_moves_to_gtp(moves: Sequence[go.PlayerMove]) -> Sequence[str]:
        return tuple(coords.to_gtp(x.move) for x in moves)

    def should_continue(self, initial_moves: Sequence[go.PlayerMove]) -> bool:
        """ client calls this to check whether it should continue the current game
        Note client might call this multiple times in a game
        """
        if len(initial_moves) < self.num_open_moves:
            return True

        gtp_moves = self._player_moves_to_gtp(initial_moves)
        key = gtp_moves[:self.num_open_moves]
        outcome = self._result_map.get(key)
        if outcome is None:
            return True

        if outcome.count == 1 and self._num_verify_games < self._max_verify_games:
            logging.info('found opening %s, rerun', ' '.join(gtp_moves))
            self._num_verify_games += 1
            return True
        logging.info('dup opening: %s, should skip', ' '.join(gtp_moves))
        return False

    def record_game(self, move_history: Sequence[go.PlayerMove], result_str):
        """ client calls this to log a finished game """
        move_history = self._player_moves_to_gtp(move_history)

        key = move_history[:self.num_open_moves]
        outcome = self._result_map.get(key)
        if outcome is None:
            self._result_map[key] = Outcome(move_history, result_str)
            return
        if outcome.moves != move_history or outcome.result != result_str:
            logging.warning('Different results for same opening: %s %s Moves=\n%s\n%s',
                            outcome.result, result_str,
                            ' '.join(outcome.moves), ' '.join(move_history))
        else:
            # dup game with the same result
            outcome.count += 1

    def record_aborted_game(self, initial_moves: Sequence[go.PlayerMove]):
        """ client log a game that's considered dup """
        gtp_moves = self._player_moves_to_gtp(initial_moves)
        assert len(gtp_moves) >= self.num_open_moves

        key = gtp_moves[:self.num_open_moves]
        assert key in self._result_map
        outcome = self._result_map[key]
        outcome.count += 1

    def to_df(self) -> pd.DataFrame:
        """ format result_map as a DataFrame """
        def format_outcome(outcome: Outcome) -> Dict:
            d = attr.asdict(outcome)
            d['moves'] = len(outcome.moves)
            d['winner'] = outcome.result[0]
            return d

        result_dict = {' '.join(k): format_outcome(v) for k, v in self._result_map.items()}
        df = pd.DataFrame.from_dict(result_dict, orient='index')
        return df

    def report(self):
        print('Tournament Stats:')
        df = self.to_df()
        print(df.sort_index())  # sort_values('count', ascending=False))
        print('Summary:\n', df['winner'].value_counts())


def test_report(argv):
    d1 = {
        ('C3', 'D3'): Outcome(tuple('abcd'), 'B+7', 3),
        ('C3', 'D2'): Outcome(tuple('abcdef'), 'W+2', 1),  # common
        ('B3', 'D2'): Outcome(tuple('abcdefg'), 'B+3', 2),
    }
    ledger1 = RedundancyChecker(2)
    ledger1._result_map = d1
    df1 = ledger1.to_df()
    d2 = {
        ('C3', 'D3'): Outcome(tuple('abcde'), 'B+5', 4),
        ('C3', 'C4'): Outcome(tuple('abcdef'), 'B+2', 2),
        ('C3', 'D2'): Outcome(tuple('abcdeg'), 'W+1', 1),  # common
    }
    ledger2 = RedundancyChecker(2)
    ledger2._result_map = d2
    df2 = ledger2.to_df()

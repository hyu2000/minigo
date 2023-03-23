""" when multiple bots analyze a game, they may differ on value estimates.
  This script resolves these disputes thru forward reasoning till they reach an agreement at a future state.
"""
import enum
import itertools
import logging

import numpy as np
import attr

import coords
import go
import k2net as dual_net
import myconf
import sgf_wrapper
from k2net import DualNetwork
from sgf_wrapper import SGFReader

logger = logging.getLogger(__name__)
MAX_GAME_LENGTH = go.N * go.N * 2


@attr.s
class Resolution:
    val1: float = attr.ib()
    val2: float = attr.ib()
    sign_final: int = attr.ib()
    num_steps_to_resolve: int = attr.ib()
    # principle variation
    pv1: str = attr.ib('')
    pv2: str = attr.ib('')

    def verdict(self):
        sign1 = np.sign(self.val1)
        sign2 = np.sign(self.val2)
        if sign1 == sign2:
            return 'AGREE'
        if sign1 == self.sign_final:
            return '1WIN'
        else:
            return '2WIN'

    def __str__(self):
        verdict = self.verdict()
        ret = f'{verdict} {self.val1:.1f} {self.val2:.1f} {self.sign_final:.0f}'
        if self.num_steps_to_resolve > 0:
            ret = f'{ret} num_steps={self.num_steps_to_resolve}'
        if self.pv1 or self.pv2:
            ret = f'{ret} pv1={self.pv1} pv2={self.pv2}'
        return ret


class DisputeResolver:
    def __init__(self, bot1: DualNetwork, bot2: DualNetwork):
        self.bot1 = bot1
        self.bot2 = bot2

    def resolve(self, pos: go.Position) -> Resolution:
        pi1, val1 = self.bot1.run(pos)
        pi2, val2 = self.bot2.run(pos)
        pos_original = pos
        val1_original, val2_original = val1, val2

        while np.sign(val1) != np.sign(val2):
            if pos.n >= MAX_GAME_LENGTH:
                # termination: should rarely happen
                benson_score = pos.score_benson()
                logger.warning(f'resolve: max game length reached, val1=%.1f, val2=%1.f, benson=%.1f, %s',
                               val1, val2, benson_score.score, 'final' if benson_score.final else 'non-final')
                val1 = val2 = benson_score.score
                break

            # disagree: whoever thinks the current player to win choose the next move
            policy_to_use = pi1 if pos.to_play * val1 > 0 else pi2
            move = np.argmax(policy_to_use * pos.all_legal_moves())

            pos = pos.play_move(coords.from_flat(move))
            pi1, val1 = self.bot1.run(pos)
            pi2, val2 = self.bot2.run(pos)

        winner_final = np.sign(val1)
        num_steps_to_resolve = pos.n - pos_original.n
        pv1 = coords.flat_to_gtp(np.argmax(pi1))
        pv2 = coords.flat_to_gtp(np.argmax(pi2))
        res = Resolution(val1_original.item(), val2_original.item(), winner_final.item(), num_steps_to_resolve,
                         pv1, pv2)
        logger.info(f'resolved pos.n={pos_original.n}: %s', res)
        return res

    def forward_consistency_check(self, pos: go.Position, bot, n_steps: int):
        """ check bot's value estimate is consistent when reasoning forward for n steps,
        taking top action from the bot
        """

    def resolve_sgf(self, sgf_fpath, out_sgf_fpath=None):
        """ go thru a game, analyze every position """
        reader = SGFReader.from_file_compatible(sgf_fpath)
        resolutions = []
        for pwc in reader.iter_pwcs():
            resolutions.append(self.resolve(pwc.position))
        final_pos = pwc.position

        if not out_sgf_fpath:
            return

        comments = [str(x) for x in resolutions[:-1]]
        logger.info(f'Writing to {out_sgf_fpath}')
        with open(out_sgf_fpath, 'w') as _file:
            sgfstr = sgf_wrapper.make_sgf(final_pos.recent,
                                          reader.result_str(),
                                          komi=final_pos.komi,
                                          game_comment=f'bot1: {self.bot1.model_id}\nbot2: {self.bot2.model_id}',
                                          comments=comments,
                                          black_name=reader.black_name(), white_name=reader.white_name())
            _file.write(sgfstr)



def setup_resolver():
    model1_id = 'model8_4'   # elo4k
    model2_id = 'model12_2'   # almost elo5k
    model1_fname = f'{myconf.EXP_HOME}/../9x9-exp2/checkpoints/{model1_id}.mlpackage'
    a0jax_fname = '/Users/hyu/PycharmProjects/a0-jax/exp-go9/tfmodel/model-218'
    model2_fname = f'{myconf.EXP_HOME}/../9x9-exp2/checkpoints/{model2_id}.mlpackage'
    bot1 = dual_net.load_net(a0jax_fname)
    bot2 = dual_net.load_net(model2_fname)
    resolver = DisputeResolver(bot1, bot2)
    return resolver


def test_resolve():
    resolver = setup_resolver()
    sgf_fpath = '/Users/hyu/Downloads/web_demo.my-end-game-loss.sgf'
    reader = SGFReader.from_file_compatible(sgf_fpath)
    # pos @ move 60
    pwc = next(itertools.islice(reader.iter_pwcs(), 60, None))
    pos0 = go.Position().play_move(coords.from_gtp('B2'))
    resolver.resolve(pos0)  #pwc.position)


def test_resolve_sgf():
    """
    seems a0jax w/o MCTS misjudges more often than elo5k w/o MCTS
    """
    resolver = setup_resolver()
    sgf_fpath = '/Users/hyu/Downloads/web_demo.my-end-game-loss.sgf'
    sgf_fpath = '/Users/hyu/Downloads/web_demo.bot-misjudge-endgame.sgf'
    out_fpath = '/Users/hyu/Downloads/resolved.sgf'
    resolver.resolve_sgf(sgf_fpath, out_fpath)

from flask import Flask
from flask import jsonify, request
from flask import redirect, url_for
import go
import web_player
import myconf
import k2net as dual_net
from absl import logging, flags
import absl.app


static_path = '/Users/hyu/PycharmProjects/dlgo/deep_learning_and_the_game_of_go/code/dlgo/httpfrontend/static'
app = Flask(__name__, static_folder=static_path)
player = None  # type: web_player.WebPlayer


def init():
    # needs flags
    global player
    model_id = 'model12_2'   # almost elo5k
    model_id = 'model8_4'   # elo4k
    model_fname = '/Users/hyu/PycharmProjects/a0-jax/exp-go9/tfmodel/model-218'
    model_fname = f'{myconf.EXP_HOME}/../9x9-exp2/checkpoints/{model_id}.mlpackage'
    dnn = dual_net.load_net(model_fname)
    logging.info('mcts %s #readouts=%d', model_id, flags.FLAGS.num_readouts)
    player = web_player.WebPlayer(dnn)


@app.route('/')
def index():
    # return redirect(url_for('static', filename='play_agents_5.html'))
    return redirect(url_for('static', filename='play_agents_9.html'))


@app.route('/select-move/<bot_name>', methods=['POST'])
def select_move(bot_name):
    content = request.json
    board_size, moves_history = content['board_size'], content['moves']
    assert board_size == go.N

    move, info = player.select_move(moves_history)

    return jsonify({
        'bot_move': move,
        'info': info
    })


def main(argv):
    init()
    app.run(port=5000, threaded=False)
    # app.run(host='0.0.0.0', port=9999, threaded=False)


if __name__ == '__main__':
    absl.app.run(main)

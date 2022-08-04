from flask import Flask
from flask import jsonify, request
from flask import redirect, url_for
import web_player
import myconf
from k2net import DualNetwork
from absl import logging, flags
import absl.app


static_path = '/Users/hyu/PycharmProjects/dlgo/deep_learning_and_the_game_of_go/code/dlgo/httpfrontend/static'
app = Flask(__name__, static_folder=static_path)
player = None


def init():
    # needs flags
    global player
    model_id = 'model10_epoch2'
    dnn = DualNetwork(f'{myconf.EXP_HOME}/models-old/model15_epoch2.h5')
    dnn = DualNetwork(f'{myconf.EXP_HOME}/checkpoints/{model_id}.h5')
    logging.info('mcts %s #readouts=%d', model_id, flags.FLAGS.num_readouts)
    player = web_player.WebPlayer(dnn)


@app.route('/')
def index():
    return redirect(url_for('static', filename='play_agents_5.html'))


@app.route('/select-move/<bot_name>', methods=['POST'])
def select_move(bot_name):
    content = request.json
    board_size, moves_history = content['board_size'], content['moves']
    assert board_size == 5

    move = player.select_move(moves_history)

    return jsonify({
        'bot_move': move,
        'diagnostics': {}
    })


def main(argv):
    init()
    app.run(port=5000, threaded=False)
    # app.run(host='0.0.0.0', port=9999, threaded=False)


if __name__ == '__main__':
    absl.app.run(main)

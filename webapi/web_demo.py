from flask import Flask
from flask import jsonify, request
from flask import redirect, url_for
import web_player
import myconf
from k2net import DualNetwork
import absl.app


static_path = '/Users/hyu/PycharmProjects/dlgo/deep_learning_and_the_game_of_go/code/dlgo/httpfrontend/static'
app = Flask(__name__, static_folder=static_path)
player = None


def init():
    # needs flags
    global player
    dnn = DualNetwork(f'{myconf.EXP_HOME}/checkpoints/model3_epoch_5.h5')
    player = web_player.WebPlayer(dnn)


@app.route('/')
def index():
    return redirect(url_for('static', filename='play_agents_9.html'))


@app.route('/select-move/<bot_name>', methods=['POST'])
def select_move(bot_name):
    content = request.json
    board_size, moves_history = content['board_size'], content['moves']
    assert board_size == 9

    move = player.select_move(moves_history)

    return jsonify({
        'bot_move': move,
        'diagnostics': {}
    })


def main(argv):
    init()
    app.run(port=5000, threaded=False)


if __name__ == '__main__':
    absl.app.run(main)

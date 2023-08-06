#!/usr/bin/python
""" this script was downloaded from
https://github.com/merrillite/puzzle2sgf

It downloads puzzles from OGS, save as sgfs.

https://apidocs.online-go.com/
"""
import requests
import os
import time

SLEEP_SECS = 0.5


# authentication is required for private problems

def escape(text):
    return text.replace('\\', '\\\\').replace(']', '\\]')


def writeInitialStones(file, string):
    for i in range(0, len(string), 2):
        file.write('[')
        file.write(string[i:i + 2])
        file.write(']')


def otherPlayer(player):
    return 'B' if player == 'W' else 'W'


def writeCoordinates(file, node):
    file.write(chr(97 + node['x']))
    file.write(chr(97 + node['y']))


def writeCoordinatesInBrackets(file, node):
    file.write('[')
    writeCoordinates(file, node)
    file.write(']')


def writeMarks(file, marks):
    for mark in marks:
        if 'letter' in mark['marks']:
            file.write('LB[')
            writeCoordinates(file, mark)
            file.write(':')
            file.write(escape(mark['marks']['letter']))
            file.write(']')
        elif 'triangle' in mark['marks']:
            file.write('TR')
            writeCoordinatesInBrackets(file, mark)
        elif 'square' in mark['marks']:
            file.write('SQ')
            writeCoordinatesInBrackets(file, mark)
        elif 'cross' in mark['marks']:
            file.write('MA')
            writeCoordinatesInBrackets(file, mark)
        elif 'circle' in mark['marks']:
            file.write('CR')
            writeCoordinatesInBrackets(file, mark)


def prependText(node, text):
    if 'text' in node:
        node['text'] = text + '\n\n' + node['text']
    else:
        node['text'] = text


def writeNode(file, node, player):
    if 'marks' in node:
        writeMarks(file, node['marks'])
    if 'correct_answer' in node:
        prependText(node, "CORRECT")
    elif 'wrong_answer' in node:
        prependText(node, "WRONG")
    if 'text' in node:
        file.write('C[')
        file.write(escape(node['text']))
        file.write(']')
    if 'branches' in node:
        branches = node['branches']
        for branch in branches:
            if len(branches) > 1:
                file.write('(')
            writeBranch(file, branch, player)
            if len(branches) > 1:
                file.write(')')


def writeBranch(file, branch, player):
    file.write(';')
    file.write(player)
    writeCoordinatesInBrackets(file, branch)
    writeNode(file, branch, otherPlayer(player))


def writePuzzle(file, puzzle):
    file.write('(;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[')
    file.write(escape(puzzle['name']))
    file.write(']SZ[')
    file.write(str(puzzle['width']))
    if puzzle['width'] != puzzle['height']:
        file.write(':')
        file.write(str(puzzle['height']))
    file.write(']')
    initial_black = puzzle['initial_state']['black']
    if initial_black:
        file.write('AB')
        writeInitialStones(file, initial_black)
    initial_white = puzzle['initial_state']['white']
    if initial_white:
        file.write('AW')
        writeInitialStones(file, initial_white)
    prependText(puzzle['move_tree'], puzzle['puzzle_description'])
    player = puzzle['initial_player'][0].upper()
    file.write('PL[')
    file.write(player)
    file.write(']')
    writeNode(file, puzzle['move_tree'], player)
    file.write(')')


def authenticate():
    url = 'https://online-go.com/api/v0/login'
    username = input('Username: ')
    password = input('Password: ')
    response = requests.post(url, data={'username': username, 'password': password})
    return response.cookies


def download_one():
    pass


def list_all_puzzles():
    url = 'https://online-go.com/api/v1/puzzles'
    puzzles = requests.get(url, cookies=[]).json()
    print(puzzles)


def list_puzzle_collections():
    url = 'https://online-go.com/api/v1/puzzles/collections'
    i_page = 0
    num_collections = 0
    while url:
        print(f'Fetching {i_page}: {url}...')
        page = requests.get(url, cookies=[]).json()
        count = page['count']
        # each page has 10 items
        results = page['results']
        for coll in results:
            puzzle_count = coll['puzzle_count']
            min_rank, max_rank = coll.get('min_rank'), coll.get('max_rank')
            starting_puzzle = coll['starting_puzzle']
            starting_puzzle_width, starting_puzzle_height = starting_puzzle['width'], starting_puzzle['height']

            if puzzle_count >= 10:
                coll_name = coll['name']
                coll_owner = coll['owner'].get('username')
                print(f"{coll_name} rank: {min_rank}-{max_rank} {puzzle_count} {coll_owner} {starting_puzzle['id']}")
                num_collections += 1

        url = page['next']
        i_page += 1
        if num_collections > 30:
            break


def download_collection(start_puzzle_id: int, out_dir: str):
    cookies = []
    url = f'https://online-go.com/api/v1/puzzles/{start_puzzle_id}/collection_summary'
    puzzle_list = requests.get(url, cookies=cookies).json()
    print('Found %d puzzles' % len(puzzle_list))
    time.sleep(SLEEP_SECS)

    for i, puzzle in enumerate(puzzle_list):
        time.sleep(SLEEP_SECS)
        url = 'https://online-go.com/api/v1/puzzles/' + str(puzzle['id'])
        response_json = requests.get(url, cookies=cookies).json()
        puzzle_json = response_json.get('puzzle')
        if puzzle_json is None:
            continue
        # puzzle_type: 'life_and_death'
        width, height = puzzle_json['width'], puzzle_json['height']
        puzzle_rank = puzzle_json.get('puzzle_rank')
        if i == 0:
            collection_json = response_json['collection']
            collection_name = collection_json['name']
            collection_dir = f'{out_dir}/{collection_name}'
            if not os.path.exists(collection_dir):
                os.mkdir(collection_dir)

        out_fname = f"{collection_dir}/{puzzle['name']}.sgf"
        with open(out_fname, 'w', encoding="utf-8") as file:
            print(f'Writing to {out_fname}: {width} {height} {puzzle_rank}')
            writePuzzle(file, puzzle_json)


def download_all():
    list_puzzle_collections()


def main():
    out_dir = '/Users/hyu/Downloads/go-puzzle9'
    downloadWholeCollection = True
    # if False, only the specified puzzle is downloaded
    # if True, all problems of the specified puzzle's collection are downloaded

    puzzleNumber = 67990
    # the puzzle id taken from the puzzle URL

    skipAuthentication = True

    cookies = [] if skipAuthentication else authenticate()
    if downloadWholeCollection:
        collectionUrl = 'https://online-go.com/api/v1/puzzles/' + str(puzzleNumber) + '/collection_summary'
        collection = requests.get(collectionUrl, cookies=cookies).json()
        print('Found %d puzzles' % len(collection))
        time.sleep(5.0)
    puzzleUrl = 'https://online-go.com/api/v1/puzzles/' + str(puzzleNumber)
    responseJSON = requests.get(puzzleUrl, cookies=cookies).json()
    if downloadWholeCollection:
        collectionName = responseJSON['collection']['name']
        collection_dir = f'{out_dir}/{collectionName}'
        os.mkdir(collection_dir)
        os.chdir(collection_dir)
    out_fname = f"{collection_dir}/{responseJSON['name']}.sgf"
    with open(out_fname, 'w', encoding="utf-8") as file:
        print(f'Writing to {out_fname}')
        writePuzzle(file, responseJSON['puzzle'])
    if downloadWholeCollection:
        for puzzle in collection:
            if puzzle['id'] != puzzleNumber:
                time.sleep(5.0)
                puzzleUrl = 'https://online-go.com/api/v1/puzzles/' + str(puzzle['id'])
                puzzleJSON = requests.get(puzzleUrl, cookies=cookies).json()['puzzle']
                out_fname = f"{collection_dir}/{puzzle['name']}.sgf"
                with open(out_fname, 'w', encoding="utf-8") as file:
                    print(f'Writing to {out_fname}')
                    writePuzzle(file, puzzleJSON)


if __name__ == '__main__':
    out_dir = '/Users/hyu/Downloads/go-puzzle9'
    # main()
    # download_all()
    download_collection(10496, out_dir)

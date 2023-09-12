""" clean/re-org puzzle collection """
import glob
import os
from typing import Optional

from sgf_wrapper import SGFReader

KANJI0 = '０'
KANJI0ORD = ord(KANJI0)
KANJI2ROMAN = str.maketrans(''.join([chr(KANJI0ORD + x) for x in range(10)]),
                            ''.join([str(x) for x in range(10)]))


def replace_all_kanji_numbers(s: str):
    return s.translate(KANJI2ROMAN)


def try_recode_puzzle_to_9x9(reader: SGFReader) -> Optional[str]:
    """ try fit a puzzle on a 9x9 board

    :return: None if impossible
    """
    return None


def batch_check_puzzles(glob_pattern: str):
    matches = sorted(glob.glob(glob_pattern))
    print('Total %d puzzles' % len(matches))
    for sgf_fname in matches:
        basename = os.path.basename(sgf_fname)
        reader = SGFReader.from_file_compatible(sgf_fname)
        print(f'{basename} {reader.board_size()} komi={reader.komi()} RE={reader.result_str()}')


def test_unicode():
    kanji1 = '１'
    kanji2 = '２'
    base0 = ord(KANJI0)
    assert ord(kanji1) == base0 + 1
    assert ord(kanji2) == base0 + 2
    print(chr(base0 + 9))

    # parens are fine
    left_paren, right_paren = '(', ')'
    assert left_paren == '('


def test_replace():
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題４級(３).sgf'
    new_fname = replace_all_kanji_numbers(sgf_fname)
    assert new_fname != sgf_fname
    assert new_fname[:-12] == sgf_fname[:-12]
    print(new_fname)
    print(len(new_fname) - len(sgf_fname))


def test_rename_files():
    """ replace Kanji numbers in filenames
    """
    # sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題９級.sgf'
    sgf_prefix = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題'
    matches = glob.glob(f'{sgf_prefix}*.sgf')
    for sgf_fname in matches:
        new_fname = replace_all_kanji_numbers(sgf_fname)
        os.rename(sgf_fname, new_fname)


def test_batch_check():
    sgf_dir = '/Users/hyu/Downloads/go-puzzle9'
    sgf_pattern = f'{sgf_dir}/Amigo no igo - 詰碁2023 - Life and Death/*'
    sgf_pattern = f'{sgf_dir}/Beginning Shapes/*'
    sgf_pattern = f'{sgf_dir}/easy 2-choice question*/*'
    sgf_pattern = f'{sgf_dir}/How to Play Go +/*'
    batch_check_puzzles(sgf_pattern)

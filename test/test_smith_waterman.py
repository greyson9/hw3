from hw3 import smith_waterman as sw
import os


def test_read_file():
    filename_a = './hw3/sequences/prot-0004.fa'
    assert sw.fa_to_seq(filename_a) == 'SLEAAQKSNVTSSWAKASAAWGTAGPEFFMALFDAHDDVF' +\
                                      'AKFSGLFSGAAKGTVKNTPEMAAQAQSFKGLVSNWVDNLD' +\
                                      'NAGALEGQCKTFAANHKARGISAGQLEAAFKVLSGFMKSY' +\
                                      'GGDEGAWTAVAGALMGEIEPDM'


def test_smith_waterman():
    alignment = sw.smith_waterman('AAA','AA', sw.read_sub_mat('./hw3/PAM100'), 13, 3)
    assert alignment[0] == 8.0
    assert alignment[1][0] == ['A', 'A']
    assert alignment[1][1] == ['A', 'A']

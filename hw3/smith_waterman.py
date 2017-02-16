# hw3.py
# implementation of the smith-waterman algorithm
import numpy as np


# read in a substitution matrix formatted in the standard way
# if a line starts with '#', ignore it
# the first line seen is the alphabet
# all lines after the alphabet are the substitution scores
def read_sub_mat(filename):
    with open(filename, 'r') as f:
        seen_letters = False
        loop = 0
        val_map = {}
        letters = []
        for line in f:
            if line[0] == '#':
                continue
            else:
                line = line.split()
                if not seen_letters:
                    seen_letters = True
                    sub_mat = {l: {} for l in line}
                    letters = line
                    val_map = dict(zip(range(len(line)), line))
                else:
                    sub_mat[val_map[loop]] = dict(zip(letters, list(map(float, line))))
                    loop += 1
    return sub_mat


#    implementation of the smith-waterman local alignment algorithm
# based on the wikipedia page, not on anyone else's code.
#     affine gap penalty
#    create two matrices: a score matrix to keep track of the score
# and a traceback matrix to keep track of where the score came from
# create the score matrix by storing the max of a first-sequence gap,
# a second-sequence gap, a substitution, or 0.
#    if anything but a 0 is stored, insert a corresponding entry
# into the traceback matrix linked to its source.
#    after completing the matrix, find the location of the max value
# then use the traceback matrix to work backwards until a 0 is reached.
#    row shifts represent a gap in sequence 1
# column shifts represent a gap in sequence 2
#    to check if a gap is an extension or an opening,
# look at the adjacent cells to see if they are the result of
# a substitution or a gap opening.  If the latter, extend.
def smith_waterman(s1, s2, sub_mat, gap_open, gap_extension):
    if len(s1) == 0 or len(s2) == 0:
        return [list(s1), list(s2)]
    score_mat = np.zeros((len(s1) + 1, len(s2) + 1))
    trace_mat = np.zeros((len(s1) + 1, len(s2) + 1))
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if trace_mat[i - 1][j] == 10:
                row_gap = score_mat[i - 1][j] - gap_extension
            else:
                row_gap = score_mat[i - 1][j] - gap_open
            if trace_mat[i][j - 1] == 1:
                col_gap = score_mat[i][j - 1] - gap_extension
            else:
                col_gap = score_mat[i][j - 1] - gap_open
            sub = score_mat[i - 1][j - 1] + sub_mat[s1[i - 1]][s2[j - 1]]
            row_sign = True
            col_sign = True
            sub_sign = True
            if row_gap <= 0:
                row_sign = False
            if col_gap <= 0:
                col_sign = False
            if sub <= 0:
                sub_sign = False
            if (not row_sign) and (not col_sign) and (not sub_sign):
                score_mat[i][j] = 0
            else:
                if sub > row_gap and sub > col_gap:
                    score_mat[i][j] = sub
                    trace_mat[i][j] = 11
                elif row_gap > col_gap:
                    score_mat[i][j] = row_gap
                    trace_mat[i][j] = 10
                else:
                    score_mat[i][j] = col_gap
                    trace_mat[i][j] = 1
    max_loc = np.argmax(score_mat)
    max_i = int(max_loc / score_mat.shape[1])
    max_j = int(max_loc % score_mat.shape[1])
    max_val = score_mat[max_i][max_j]
    alignment = [[], []]
    while max_val != 0:
        if trace_mat[max_i][max_j] == 11:
            alignment[0].append(s1[max_i - 1])
            alignment[1].append(s2[max_j - 1])
            max_i -= 1
            max_j -= 1
        elif trace_mat[max_i][max_j] == 10:
            alignment[0].append(s1[max_i - 1])
            alignment[1].append('-')
            max_i -= 1
        elif trace_mat[max_i][max_j] == 1:
            alignment[0].append('-')
            alignment[1].append(s2[max_j - 1])
            max_j -= 1
        max_val = score_mat[max_i][max_j]
    alignment[0].reverse()
    alignment[1].reverse()
#     print(trace_mat)
#     print(score_mat)
    return np.amax(score_mat), alignment


# calculate the local alignment score threshold which gives
# a TPR as close to 0.7 as possible
# return the score threshold, the difference of the TPR from 0.7
def find_tp_thresh(pos_arr, neg_arr):
    thresh = 0
    diff = 0.5
    best_tpr = 1.0
    best_fpr = 1.0
    range_min = int(min(min(pos_arr), min(neg_arr)))
    range_max = int(max(max(pos_arr), max(neg_arr)))
    for i in range(range_min, range_max):
        true_pos = len(pos_arr[pos_arr > i])
        tpr = true_pos / len(pos_arr)
        if np.absolute(tpr - 0.7) < diff:
            diff = np.absolute(tpr - 0.7)
            thresh = i
            best_fpr = len(neg_arr[neg_arr > i]) / len(neg_arr)
            best_tpr = tpr
    print(best_fpr)
    print(best_tpr)
    return thresh, diff, best_fpr, best_tpr

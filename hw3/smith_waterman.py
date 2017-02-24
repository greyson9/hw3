# hw3.py
# implementation of the smith-waterman algorithm
import numpy as np
import itertools
import time
import string


# read in the contents of a .fa file and return the sequence
def fa_to_seq(filename):
    out = ''
    with open(filename, 'r') as f:
        for line in f:
            if line[0] != '>':
                out += line.strip()
    out = out.translate({ord(i):None for i in 'x'})
    return out


# read in a pair file and return a list of associated filenames
def pair_to_fn(filename):
    out_mat = []
    with open(filename, 'r') as f:
        for line in f:
            _ = line.split()
            out_mat.append(('./hw3/'+_[0], './hw3/'+_[1]))
    return out_mat


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
# look at the traceback matrix corresponding to the adjacent cells
# to see if they are the result of a substitution or a gap opening.
# If the latter, extend. Else, open a gap.
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
    return np.amax(score_mat), alignment
#     return np.amax(score_mat)/min(len(s1), len(s2)), alignment


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


# Use simulated annealing to optimize substitution matrix
# A candidate matrix is the last selected matrix with entries increased by a random quantity
# between 0 and 1, if there is a higher frequency of substitution for the corresponding
# amino acids in the positive alignment set than the negative alignment set.
#
# At each step of the annealing schedule, construct a candidate matrix
# Compute the scores of the old and new matrices (sum of TPR for FPRS of 0, 0.1, 0.2, 0.3)
# If the score of the new matrix is greater than the old matrix, set the old matrix to the new one and iterate
# If the score of the new matrix is less than the old matrix, calculate prob = exp((new - old) / temp)
# If prob > a random draw between 0 and 1, accept the new matrix and iterate
# Otherwise, keep the old matrix and iterate
def optimize_matrix(start_mat, mat_to_opt):
    pos_aligns = [smith_waterman(fa_to_seq(x[0]), fa_to_seq(x[1]), start_mat, 13, 3)[1] for x in pair_to_fn('./hw3/Pospairs.txt')]
    neg_aligns = [smith_waterman(fa_to_seq(x[0]), fa_to_seq(x[1]), start_mat, 13, 3)[1] for x in pair_to_fn('./hw3/Negpairs.txt')]
    freqs = {x: {y: [0, 0, 0] for y in read_sub_mat('./hw3/PAM100').keys()} for x in read_sub_mat('./hw3/PAM100').keys()}
    for align in pos_aligns:
        s1 = align[0]
        s2 = align[1]
        for char1, char2 in zip(s1, s2):
            if char1 != '-' and char2 != '-':
                freqs[char1][char2][0] += 1
                freqs[char2][char1][0] += 1
    for align in neg_aligns:
        s1 = align[0]
        s2 = align[1]
        for char1, char2 in zip(s1, s2):
            if char1 != '-' and char2 != '-':
                freqs[char1][char2][1] += 1
                freqs[char2][char1][1] += 1
    for a1 in freqs.keys():
        pos_subs = sum([freqs[a1][x][0] for x in freqs[a1].keys()])
        neg_subs = sum([freqs[a1][x][1] for x in freqs[a1].keys()])
        for a2 in freqs.keys():
            if pos_subs > 0:
                if neg_subs > 0:
                    freqs[a1][a2][2] = freqs[a1][a2][0] / pos_subs - freqs[a1][a2][1] / neg_subs
                    freqs[a2][a1][2] = freqs[a1][a2][2]
                else:
                    freqs[a1][a2][2] = freqs[a1][a2][0] / pos_subs
                    freqs[a2][a1][2] = freqs[a1][a2][2]



    old_mat = mat_to_opt
    sorted_pos = np.sort(np.asarray([smith_waterman(fa_to_seq(x[0]), fa_to_seq(x[1]), old_mat, 13, 3)[0] for x in pair_to_fn('./hw3/Pospairs.txt')]))
    sorted_neg = np.sort(np.asarray([smith_waterman(fa_to_seq(x[0]), fa_to_seq(x[1]), old_mat, 13, 3)[0] for x in pair_to_fn('./hw3/Negpairs.txt')]))
    old_score = (len(sorted_pos[sorted_pos > sorted_neg[49]]) + len(sorted_pos[sorted_pos > sorted_neg[44]])
                 + len(sorted_pos[sorted_pos > sorted_neg[39]]) + len(sorted_pos[sorted_pos > sorted_neg[34]])
                ) / len(sorted_pos)
    # warming then cooling gives annealing schedule
    temp_sched = [4 * ((1.1) ** n) for n in range(0, 10)]
    temp_sched.extend([temp_sched[-1] * ((0.95) ** n) for n in range(0, 1000)])
    # testing annealing
    # temp_sched = [9 * ((1.1) ** n) for n in range(0, 2)]
    # temp_sched.extend([temp_sched[-1] * ((0.9) ** n) for n in range(0, 2)])
    for temp in temp_sched:
        # adjust each member of old matrix by random number, with sign derived from static aligned sequences
        new_mat = {x: {
            y: old_mat[x][y] + np.sign(freqs[x][y][2]) * np.random.rand(1, 1)[0][0] for y in old_mat[x].keys()
        } for x in old_mat.keys()}

        sorted_pos = np.sort(np.asarray([smith_waterman(
            fa_to_seq(x[0]), fa_to_seq(x[1]), new_mat, gap_open, gap_extend)[0] for x in pair_to_fn('./hw3/Pospairs.txt')]))
        sorted_neg = np.sort(np.asarray([smith_waterman(
            fa_to_seq(x[0]), fa_to_seq(x[1]), new_mat, gap_open, gap_extend)[0] for x in pair_to_fn('./hw3/Negpairs.txt')]))
        new_score = (len(sorted_pos[sorted_pos > sorted_neg[49]]) + len(sorted_pos[sorted_pos > sorted_neg[44]])
                     + len(sorted_pos[sorted_pos > sorted_neg[39]]) + len(sorted_pos[sorted_pos > sorted_neg[34]])
                    ) / len(sorted_pos)
        prob = min(1, np.exp(-(old_score - new_score) / temp))
        if prob > np.random.rand(1, 1)[0][0]:
            old_score = new_score
            old_mat = new_mat
    return old_mat

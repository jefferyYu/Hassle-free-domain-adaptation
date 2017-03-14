# PyTC Functions V3.30, last updated on 2013-03-13.

import os, re, random, math
import numpy as np

LOG_LIM = 1E-300

liblinear_learn_exe = 'liblinear/train'
liblinear_classify_exe = 'liblinear/predict'

########## Sample Building ##########

def build_samps(term_dict, class_dict, doc_terms_list, doc_class_list, term_weight, idf_term = None):
    '''
    New functions for building samples to sparse format from term list, 2010-3-23
    term_dict, for example, term1: 1; term2:2; term3:3, ...
    class_dict, for example, negative:1; postive:2; unlabel:0 
    '''
    samp_dict_list = []
    samp_class_list = []
    for k in range(len(doc_class_list)):
        doc_class = doc_class_list[k]
        samp_class = class_dict[doc_class]
        samp_class_list.append(samp_class)
        doc_terms = doc_terms_list[k]
        samp_dict = {}
	for term in doc_terms:
	    if term_dict.has_key(term):
		term_id = term_dict[term]
		if term_weight == 'BOOL':
		    samp_dict[term_id] = 1
		elif term_weight == 'TF':
		    if samp_dict.has_key(term_id):
			samp_dict[term_id] += 1
		    else:
			samp_dict[term_id] = 1
		elif term_weight == 'TFIDF':
		    if samp_dict.has_key(term_id):
			samp_dict[term_id] += idf_term[term]
		    else:
			samp_dict[term_id] = idf_term[term]
        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

def samp_length_norm(samp_dict_list):
    for samp_dict in samp_dict_list:
        sum = 0.0
        for i in samp_dict:
            sum += samp_dict[i]
        for j in samp_dict:
            samp_dict[j] /= sum

def save_samps(samp_dict_list, samp_class_list, fname, feat_num = 0):
    length = len(samp_class_list)
    fout = open(fname, 'w')
    for k in range(length):
        samp_dict = samp_dict_list[k]
        samp_class = samp_class_list[k]
        fout.write(str(samp_class) + '\t')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()
    
def save_samps_unlabel(samp_dict_list, fname, feat_num = 0):
    length = len(samp_dict_list)
    fout = open(fname, 'w')    
    for k in range(length):
        samp_dict = samp_dict_list[k]
        #fout.write('0\t')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()    

def load_samps(fname, fs_num = 0):
    fsample = open(fname, 'r')
    samp_class_list = []
    samp_dict_list = []
    for strline in fsample:
        samp_class_list.append(strline.strip().split()[0])
        if fs_num > 0:
            samp_dict = dict([[int(x.split(':')[0]), float(x.split(':')[1])] for x in strline.strip().split()[1:] if int(x.split(':')[0]) < fs_num])
        else:
            samp_dict = dict([[int(x.split(':')[0]), float(x.split(':')[1])] for x in strline.strip().split()[1:]])
        samp_dict_list.append(samp_dict)
    fsample.close()
    return samp_dict_list, samp_class_list


########## Classification Functions ##########

def liblinear_exe(fname_samp_train, fname_samp_test, fname_model, fname_output, learn_opt = ' ', classify_opt = ' '):
    print '\nLiblinear executive classifing...'
    os.system(liblinear_learn_exe + ' ' +  learn_opt + ' ' + fname_samp_train + ' ' + fname_model)
    os.system(liblinear_classify_exe + ' ' + classify_opt + ' ' + fname_samp_test + ' ' + fname_model + ' ' + fname_output)
    samp_class_list_test = [x.split()[0] for x in open(fname_samp_test).readlines()]
    samp_class_list_svm = [x.split()[0] for x in open(fname_output).readlines()[1:]]
    acc = calc_acc(samp_class_list_svm, samp_class_list_test)
    return acc

########## Evalutation Functions ##########

def calc_acc(labellist1, labellist2):
    if len(labellist1) != len(labellist2):
        print 'Error: different lenghts!'
        return 0
    else:
        samelist =[int(x == y) for (x, y) in zip(labellist1, labellist2)]
        acc = float((samelist.count(1)))/len(samelist)
        return acc
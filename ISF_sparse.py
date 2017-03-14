from numpy import *  
import operator  
import re
import pytc
import os
import numpy as np
import random
import time, logging
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger("da.isf")

def fixed_shuffle(li, seed=0):
    a = random.Random(seed)
    a.shuffle(li)
    
def read_fbyline(fname_list):
    cls_list = []
    doc_list = []
    term_dict = {}
    fin = open(fname_list,'r')	
    for str_line in fin.readlines():
	samp_dict = {}
	doc_str = str_line.split(' ')[0]
	cls_list.append(doc_str)
	key_value_list = str_line.split(' ')[1:]
	for key_value in key_value_list:
	    if key_value != '\n':
		key = int(key_value.split(':')[0])
		value = int(key_value.split(':')[1])
		samp_dict[key] = value
		if not term_dict.has_key(key):
		    term_dict[key] = 1
		elif term_dict.has_key(key):
		    term_dict[key] += 1
	doc_list.append(samp_dict)		
    
    return doc_list, cls_list, term_dict

def save_new_samps(samp_dict_list, samp_class_list, fname, allhx, set_len, feat_num = 0):
    length = len(samp_class_list)
    fout = open(fname, 'w')
    term_set_len = set_len
    for k in range(length):
	samp_dict = samp_dict_list[k]
	samp_class = samp_class_list[k]
	fout.write(str(samp_class) + ' ')
	for term_id in sorted(samp_dict.keys()):
		if feat_num == 0 or term_id < feat_num:
			fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
	for i in range(len(allhx[k])):
		fout.write(str(term_set_len+i+1) + ':' + str(allhx[k][i]) + ' ')
	fout.write('\n')
    fout.close()	

def learn_representation(train_data, test_data):	
    data = []
    rows = []
    cols = []
    fea_idx = 0
    inst_idx = 0
    for instance in train_data:
	samp_dict = instance
	key_list = samp_dict.keys()	
        for feature in key_list:
            data.append(1.0)
            cols.append(inst_idx)
	    rows.append(feature-1)
        inst_idx = inst_idx + 1

    for instance in test_data:
	samp_dict = instance
	key_list = samp_dict.keys()	
        for feature in key_list:
            data.append(1.0)
            cols.append(inst_idx)
	    rows.append(feature-1)
        inst_idx = inst_idx + 1

    xx = sparse.csc_matrix((data,(rows,cols)))
    #print(xx[0])
    #print(xx.shape)
    xx = np.transpose(xx)
    #print(xx[0])
    #print(xx.shape)
    
    return xx

class ISF:
    """
    Implements unsupervised domain adaptation method using instance similarity features
    """	
    def __init__(self, source, target, output):
	self.source_domain = source
	self.target_domain = target
	self.output_dir = output
    
    def domainadap(self, center_list, K):
	foldnum = 5  # five-cross validation
	task_dir = self.output_dir + os.sep + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0]
	if not os.path.exists(task_dir):
	    os.mkdir(task_dir)

	fname_input_train = self.source_domain
	fname_input_test = self.target_domain				

	
	print 'Building samples...'
    
	samp_list_train_total, train_cls_total, train_term_dict = read_fbyline(fname_input_train)			
	samp_list_test, test_cls_total, test_term_dict = read_fbyline(fname_input_test)		

	for fold in xrange(foldnum):	#xrange(foldnum)
	    acc_list2 = []
	    fname_samps_test = task_dir + os.sep +  self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + 'test.samp' + str(fold+1)
	    fname_samps_train = task_dir + os.sep +  self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + 'train.samp' + str(fold+1)				
	    fname_term_set = task_dir + os.sep +  self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + 'term.set' + str(fold+1)
	    fname_train_sample = task_dir + os.sep + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + 'trainsample' + str(fold+1)
	    fname_test_sample = task_dir + os.sep + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] +'testsample' + str(fold+1) 
	    fname_model_libsvm = task_dir + os.sep  + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + '.model' + str(fold+1)
	    fname_output_libsvm = task_dir + os.sep  + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + '.output' + str(fold+1)  	 
	    fname_newtrain_sample1 = task_dir + os.sep + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + 'newtrainsample' + str(fold+1)
	    fname_newtest_sample1 = task_dir + os.sep + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] +'newtestsample'  + str(fold+1) 				
	    fname_newmodel_libsvm1 = task_dir + os.sep  + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + '.newmodel' + str(fold+1)
	    fname_newoutput_libsvm1 = task_dir + os.sep  + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + '.newoutput' + str(fold+1)			
	    fname_comp = task_dir + os.sep  + self.source_domain.split('.')[0] + '2' + self.target_domain.split('.')[0] + '.comp'  + str(fold+1)
	    
	    # conduct five-fold cross-validation on the source data(i.e., training data), that is each time we use four folds from the source domain as training data, and all samples in the target domain as test data
	    batch_size = len(samp_list_train_total)/foldnum	
	    if fold == 0:
		samp_list_train = samp_list_train_total[batch_size:]
		train_cls_fold = train_cls_total[batch_size:]
	    elif fold == 4:
		samp_list_train = samp_list_train_total[:batch_size*4]
		train_cls_fold = train_cls_total[:batch_size*4]			
	    else:
		samp_list_train = samp_list_train_total[:batch_size*fold] + samp_list_train_total[batch_size*(fold+1):]	
		train_cls_fold = train_cls_total[:batch_size*fold] + train_cls_total[batch_size*(fold+1):]				
	    
	    #feature sets of target and source domain	
	    train_term_set = train_term_dict.keys()
	    test_term_set = test_term_dict.keys()
	    #union of the feature sets
	    term_set = list(set(train_term_set).union(set(test_term_set)))	
	    set_len = max(term_set)	# max(term_set) means the length of the feature sets, +1 means the starting feature position of our appended features 		
	    
	    print 'save baseline samples'
	    xx = learn_representation(samp_list_train, samp_list_test)
	    pytc.save_samps(samp_list_train, train_cls_fold, fname_samps_train)
	    pytc.save_samps(samp_list_test, test_cls_total, fname_samps_test)		
	    
	    for itera in xrange(len(center_list)):
		center_num = center_list[itera]
		length = len(samp_list_train) 
		testlength = len(samp_list_test)
		xx_test = xx[length:]
		perm = np.random.permutation(testlength)
		xx_test = xx_test[perm]
		test_center = xx_test[:center_num]  
    
		print 'transform samples'
		norm_test_center = preprocessing.normalize(test_center)
		allhx = linear_kernel(xx, norm_test_center)
		#U, S, V = np.linalg.svd(allhx, False, True)
		#allhx = U[:, :K]		
	
		save_new_samps(samp_list_train, train_cls_fold, fname_newtrain_sample1, allhx[:length,:], set_len)
		save_new_samps(samp_list_test, test_cls_total, fname_newtest_sample1, allhx[length:,:], set_len)  	    
				
		learn_opt = '-s 7 -c  ' + str(1)
		classifly_opt = '-b 1'                

		if itera == 0:
		    print 'classification for baseline'
		    acc = pytc.liblinear_exe(fname_samps_train, fname_samps_test, fname_model_libsvm, fname_output_libsvm, learn_opt, classifly_opt) 
		
		print 'classification for original features plus dot product features'
		acc2 = pytc.liblinear_exe(fname_newtrain_sample1, fname_newtest_sample1, fname_newmodel_libsvm1, fname_newoutput_libsvm1, learn_opt, classifly_opt) 
		acc_list2.append(acc2)			
	    fout = open(fname_comp, 'w')
	    fout.write('original : acc='+ str(acc)+ '\n')

	    for i in xrange(len(acc_list2)):
		fout.write('new : ' + str(center_list[i]) + 'acc='+ str(acc_list2[i])+ '\n')			
	    fout.close()		
		
def main():
    output_dir = 'result' 
    source_domain = 'train.feat'        # source domain data
    target_domain = 'u01.feat'      # target domain data

    if not os.path.exists(output_dir):
	os.mkdir(output_dir)
    
    center_list = [100]  #the number of smaples chosen from the target domain  [10, 20, 50, 100, 200, 500, 1000]
    
    isf = ISF(source_domain, target_domain, output_dir)
    K = 25
    isf.domainadap(center_list, K)
	    
if __name__ == '__main__':   
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')     
    main()
    logger.info('end logging')
				

			


	
	
	
	

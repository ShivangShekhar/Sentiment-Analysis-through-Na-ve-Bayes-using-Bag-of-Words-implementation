import re
import pandas as pd
from nltk.tokenize import word_tokenize 
import numpy as np


accuracy=np.zeros([5,1])
f1_score=np.zeros([5,1])
for fold in range(0,5):
    tp=0
    tn=0
    fp=0
    fn=0

    file  = open('a1_d3.txt')
    file1 = open('good.txt','w')
    file2 = open('bad.txt','w')
    file3 = open('test.txt','w')
    good = 0
    bad  = 0
    line_count = 0

    
    while True:
        line = file.readline()
        if not line:
            break
        line_count = line_count+1
     
        line = re.sub('[^a-z\s10]+',' ',line, flags=re.IGNORECASE) #every char except alphabets is replaced
        line = re.sub('(\s+)',' ',line) #multiple spaces are replaced by single space
        line = line.lower() #converting the cleaned string to lower case
        line = line + ' '
        temp = ""
        words = word_tokenize(line)
        for j in words:
    
            temp = temp + j
            temp = temp + ' '
        if line_count>=200*fold and line_count<200*(fold+1):
            file3.writelines(temp.strip())
            file3.write("\n")
        else:
            length=len(temp)
            if temp[-2]=='1':
                good += 1
                new_str = temp[:length-2]
                new_str = new_str+' '
                file1.writelines(new_str)
            elif temp[-2]=='0':
                bad += 1
                new_str = temp[:length-2] 
                new_str = new_str+' '
                file2.writelines(new_str)
    good_list = open('good.txt')
    line = good_list.readline()
    wordlist = line.split()
    wordfreq = [wordlist.count(p) for p in wordlist]
    freq_good = dict(zip(wordlist,wordfreq))
    g_data = pd.DataFrame.from_dict(freq_good, orient = 'index').reset_index()
        
    bad_list = open('bad.txt')
    line=bad_list.readline()
    wordlist = line.split()
    wordfreq = [wordlist.count(p) for p in wordlist]
    freq_bad = dict(zip(wordlist,wordfreq))
    b_data = pd.DataFrame.from_dict(freq_bad, orient = 'index').reset_index()
    
    total_good = g_data[0].sum()
    total_bad  = b_data[0].sum()
    p_good = good/(good+bad)
    p_bad = bad/(good+bad)
    vocab_length = len(freq_good)+len(freq_bad)
    
    correct=0
    
    tc = 0 #test_cases
    testing = open('test.txt')
    tp=0
    tn=0
    fp=0
    fn=0
    while True:
        test_query = testing.readline()
        if not test_query:
            break
        tc = tc+1
        length = len(test_query)
        senti = test_query[(length-2)]
        query = test_query[:length-2]
        wordlist = word_tokenize(query)
        pro_good = 1
        pro_bad  = 1

        for a in wordlist:
            word = a
    
            if word in freq_good:
                count_good = freq_good[word]
            else:
                count_good = 0
            if word in freq_bad:
                count_bad = freq_bad[word]
            else:
                count_bad=0
            prob_good = (count_good+1)/(total_good+vocab_length+1)
            prob_bad = (count_bad+1)/(total_bad+vocab_length+1)
            pro_good *=prob_good
            pro_bad *=prob_bad
        pro_good *= good/(good+bad)
        pro_bad  *= bad/(good+bad)
        
        if pro_good>pro_bad:
            pred = '1'
        else:
            pred = '0'
        if pred == senti:
            correct +=1
        if(pred=='1' and senti=='1'):
            tp +=1
        if(pred=='1' and senti=='0'):
            fp +=1
        if(pred=='0' and senti=='0'):
            tn +=1
        if(pred=='0' and senti=='1'):
            fn +=1
    accuracy[fold,0] = correct/tc
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score[fold,0] = 2*precision*recall/(precision+recall)
    
mean = round(np.mean(accuracy),4)
dev = round(np.std(accuracy),4)
#precision = tp/(tp + fp)
#recall = tp/(tp + fn)
#f1_score = 2*precision*recall/(precision+recall)
print("Accuracy = ", 100*mean,"% +/- ", 100*dev,"%") 
print("Precision : ",precision)
print("Recall : ",recall)
mean = round(np.mean(f1_score),4)
dev = round(np.std(f1_score),4)
print("F-Score : ", mean," +/- ", dev) 
        
        

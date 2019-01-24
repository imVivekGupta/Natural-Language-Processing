import re, sys
import copy
import math
from nltk.corpus import brown
from collections import Counter
from nltk import bigrams, ngrams, trigrams
import numpy as np

print('Getting Brown Data ...', end='\t')
data = brown.sents()[:40000]
print('Done.')

def preprocess(data):
    processed_data = []
    for sentence in data:
        new_sent = []
        for word in sentence:
            new_word = ''.join(re.findall('[a-z]+',word.lower()))
            if(new_word != ''):
                new_sent.append(new_word)
        if(len(new_sent)!=0):
            processed_data.append(new_sent)
    return processed_data        

def MLE_perplexity(prob,test_sent):
    if(prob==0):
        return ('Log-likelihood: -inf | Perplexity: inf')
    N = len(test_sent.split())
    return ('Log-likelihood: {:0.4f} | Perplexity: {:0.4f}'.format(math.log(prob),pow(prob,-1/N)))

test_sentences = []
outfile = open('./output.txt','w')
try:
    infile = sys.argv[1]
    with open(infile,'r') as infile:
        print('Fetching Test sentences ... ', end='\t')
        test_sentences = [line.split() for line in infile.readlines()]
        print('Done.')
except:
    print('specify valid path to input test file as: python Assignment_1_15CS10053 <path>')
    exit(0)

test_sentences = preprocess(test_sentences)
test_sentences = [' '.join(sent) for sent in test_sentences]

unigrams = []
processed_data = preprocess(data)
for sent in processed_data:
    unigrams.extend(sent)
    
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)
unigram_model = {}

print('creating unigram model ...',end='\t')
for word in unigram_counts:
    unigram_model[word] = unigram_counts[word]/unigram_total
print('Done')

def get_bigrams(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent,2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1     
    return model

bigram_counts= get_bigrams(processed_data)
bigram_model = copy.deepcopy(bigram_counts)

print('creating bigram model ...',end='\t')
for w1 in bigram_model:
        tot_count=float(sum(bigram_model[w1].values()))
        for w2 in bigram_model[w1]:
            bigram_model[w1][w2]/=tot_count
print('Done')

def get_trigrams(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent,3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1     
    return model

trigram_counts= get_trigrams(processed_data)
trigram_model = copy.deepcopy(trigram_counts)

print('creating trigram model ...',end='\t')
for (w1,w2) in trigram_model:
        tot_count=float(sum(trigram_model[(w1,w2)].values()))
        for w3 in trigram_model[(w1,w2)]:
            trigram_model[(w1,w2)][w3]/=tot_count
print('Done')

import matplotlib.pyplot as plt
def zipf_plot(sorted_ngrams,ngram):
    x_axis=[ math.log(i) for i in range(1,len(sorted_ngrams)+1)]
    y_axis= [ math.log(i[1]) for i in sorted_ngrams]
    plt.clf()
    plt.xlabel('Log rank')
    plt.ylabel('Log frequency')
    plt.title('Zipf\'s law for {}'.format(ngram))
    plt.plot(x_axis,y_axis)
    plt.savefig('./zipf_{}.png'.format(ngram))

print('Plotting zipf law for unigrams ...',end='\t')
sorted_unigrams = unigram_counts.most_common()
zipf_plot(sorted_unigrams,'unigrams')
print('Done')

print('Plotting zipf law for bigrams ...',end='\t')
bigram_freq = []
for w1 in bigram_counts:
    for w2 in bigram_counts[w1]:
        bigram_freq.append([(w1,w2),bigram_counts[w1][w2]])

sorted_bigrams = sorted(bigram_freq , key = lambda x: x[1], reverse=True)
zipf_plot(sorted_bigrams,'bigrams')
print('Done')
print('Plotting zipf law for trigrams ...',end='\t')
trigram_freq = []
for (w1,w2) in trigram_counts:
    for w3 in trigram_counts[(w1,w2)]:
        trigram_freq.append([(w1,w2,w3),trigram_counts[(w1,w2)][w3]])

sorted_trigrams = sorted(trigram_freq , key = lambda x: x[1], reverse=True)
zipf_plot(sorted_trigrams,'trigrams')
print('Done')

actual_sorted_bigrams = [bg for bg in sorted_bigrams if (bg[0][0]!=None and bg[0][1]!=None)]
actual_sorted_trigrams = [tg for tg in sorted_trigrams if(not(tg[0][0]==None or tg[0][1]==None or tg[0][2]==None))]

print('printing top ten n-grams ...',end='\t')
# top ten unigrams, bigrams, trigrams
print('Task 1 : N-gram models without smoothing\n', file=outfile)
print('\nTop ten unigrams:', file=outfile)
print(sorted_unigrams[:10], file=outfile)
print('\nTop ten bigrams:', file=outfile)
print(actual_sorted_bigrams[:10], file=outfile)
print('\nTop ten trigrams:', file=outfile)
print(actual_sorted_trigrams[:10], file=outfile)
print('Done')

print('Testing on test data ... ',end='\t')
print('\nUnigram test probabilities', file=outfile)
for elem in test_sentences:
    p_val=np.prod([unigram_model[i] for i in elem.split()])
    print('Sequence: {:20} | Unigram prob: {:0.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)

print('\nBigram test probabilities', file=outfile)

for elem in test_sentences:
    p_val=1
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        if(w1 in bigram_model):
            if(w2 in bigram_model[w1]):
                p_val*=bigram_model[w1][w2]
                continue
        p_val = 0
        break
    print('Sequence: {:20} | Bigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)
    

print('\nTrigram test probabilities', file=outfile)
for elem in test_sentences:
    p_val=1
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val*=trigram_model[(w1,w2)][w3]
        except Exception as e:
            p_val=0
            break
    print('Sequence: {:20} | Trigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)
print('Done')

# Laplacian smoothing

k = [1e-4, 1e-3, 1e-2, 0.1, 1]
print('###########################################', file=outfile)
print('Task 2 : Laplacian Smoothing', file=outfile)
print('Laplacian smoothing ...')
for alpha in k:
    print('k={}'.format(alpha))
    print('\nk = {} | Unigram model:'.format(alpha), file=outfile)
    # unigrams
    N = len(sorted_unigrams)     # unseen test unigrams not taken into account
    for elem in test_sentences:
        p_val = 1
        for w1 in elem.split():
            if(w1 in unigram_model):
                p_val *= (unigram_counts[w1]+alpha)/(alpha*N + unigram_total)
            else:
                p_val *= alpha/(alpha*N + len(unigram_model))
        print('Sequence: {:20} | Unigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)

    # bigrams
    print('\nk = {} | Bigram model:'.format(alpha), file=outfile)
   
    for elem in test_sentences:
        p_val=1
        for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
            if(w1 in bigram_counts):
                t = sum(bigram_counts[w1].values())
                if(w2 in bigram_counts[w1]):
                    p_val *=  (alpha + bigram_counts[w1][w2])/(alpha*N + t)
                else:
                    p_val *= alpha/(alpha*N + t)
            else:
                p_val *= 1/N    
        print('Sequence: {:20} | Bigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)
    
    # trigrams
    print('\nk = {} | Trigram model:'.format(alpha), file=outfile)
    #N = N**2
    for elem in test_sentences:
        p_val=1
        for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
            if((w1,w2) in trigram_counts):
                t = sum(trigram_counts[(w1,w2)].values())
                if(w3 in trigram_counts[(w1,w2)]):
                    p_val *= (alpha + trigram_counts[(w1,w2)][w3])/(alpha*N + t)
                else:
                    p_val *= alpha/(alpha*N + t)
            else:
                p_val *= 1/N
        print('Sequence: {:20} | Trigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)

print('Good turing smoothing ...')
# good turing
print('###########################################', file=outfile)
print('Task 3 : Good Turing Smoothing\n', file=outfile)
total_possible_bigrams = (len(unigram_counts)+1)**2 # 1 to account for None
deno = sum([i[1] for i in sorted_bigrams])
n = {0:total_possible_bigrams-len(sorted_bigrams)}
rev_bigrams = list(reversed(sorted_bigrams))
for bg in rev_bigrams:
    if(bg[1] not in n):
        n[bg[1]] = 1
    else:
        n[bg[1]] += 1
        
counts = list(n.keys())
for rank in range(0,len(counts)-1):
    n[counts[rank]] = ((rank+1)*n[counts[rank+1]])/(n[counts[rank]]*deno)
n[counts[-1]] = bigram_model[sorted_bigrams[0][0][0]][sorted_bigrams[0][0][1]]
pGT_bigram = {None:{None:n[0]}}
for bg in rev_bigrams:
    if(bg[0][0] in pGT_bigram):
        pGT_bigram[bg[0][0]][bg[0][1]] = n[bg[1]]
    else:
         pGT_bigram[bg[0][0]] = {bg[0][1]: n[bg[1]]}
            
print('Bigram model\n', file=outfile)
for elem in test_sentences:
    p_val=1
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        if(w1 in pGT_bigram):
            if(w2 in pGT_bigram[w1]):
                p_val*=pGT_bigram[w1][w2]
                continue
        p_val *= pGT_bigram[None][None]
    print('Sequence: {:20} | Bigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)

total_possible_trigrams = (len(unigram_counts)+1)**3 # 1 to account for None
n = {0:total_possible_trigrams-len(sorted_trigrams)}
deno = sum([i[1] for i in sorted_trigrams])
rev_trigrams = list(reversed(sorted_trigrams))
for tg in rev_trigrams:
    if(tg[1] not in n):
        n[tg[1]] = 1
    else:
        n[tg[1]] += 1
        
counts = list(n.keys())
for rank in range(0,len(counts)-1):
    n[counts[rank]] = ((rank+1)*n[counts[rank+1]])/(n[counts[rank]]*deno)
n[counts[-1]] = trigram_model[(sorted_trigrams[0][0][0],sorted_trigrams[0][0][1])][sorted_trigrams[0][0][2]]
pGT_trigram = {(None,None):{None:n[0]}}
for tg in rev_trigrams:
    if((tg[0][0],tg[0][1]) in pGT_trigram):
        pGT_trigram[(tg[0][0],tg[0][1])][tg[0][2]] = n[tg[1]]
    else:
         pGT_trigram[(tg[0][0],tg[0][1])] = {tg[0][2]: n[tg[1]]}
            
print('\nTrigram model\n', file=outfile)
for elem in test_sentences:
    p_val=1
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        if((w1,w2) in pGT_trigram):
            if(w3 in pGT_trigram[(w1,w2)]):
                p_val*=pGT_trigram[(w1,w2)][w3]
                continue
        p_val *= pGT_trigram[(None,None)][None]
    print('Sequence: {:20} | Trigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)

# interpolation
print('Interpolation ...')
lamda = [0.2,0.5,0.8]
print('###########################################', file=outfile)
print('Task 4 : Interpolation Method', file=outfile)
for l in lamda:
    print('lambda = {}'.format(l))
    print('\nLambda : {}'.format(l), file=outfile)
    for elem in test_sentences:
        p_val = 1 
        for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
            p_bigram = 0        # for unseen bigrams
            p_unigram = 0
            if(w1 in bigram_model):
                if(w2 in bigram_model[w1]):
                    p_bigram = bigram_model[w1][w2]
            if(w2 in unigram_model):
                p_unigram = unigram_model[w2]
            p_val *= (l*p_bigram + (1-l)*p_unigram)
        print('Sequence: {:20} | Bigram prob: {:.4e}| {}'.format(elem,p_val,MLE_perplexity(p_val,elem)), file=outfile)
print('Completed.')
outfile.close() 
import numpy as np
import math
import os
import sys
import runpy


#beginning bpe

with open('input.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

vocab = {}
for w in raw_text.split():
    w = " ".join(list(w)) + " </w>" #ok so basically right here we are taking each letter of the string as evidenced by the list() and then putting spaces in between each character and putting an end of word token at the end
    if w in vocab:
        vocab[w] += 1 #counting frequency of words in the vocab dict so we can easily find out later what the most popular pair is to make.
    else:
        vocab[w] = 1

ps = {} #short for pairs
def pairs():
    ps.clear()
    for w, f in vocab.items(): #basically we are splitting the vocab dictionary into words and frequencies again so now we can make pairs that will come with their frequency so i can use greedy sorting to simply merge the highest frequency one
        chars = w.split() #the spaces are coming in handy. basically we just seperated it into characters again
        for i in range(len(chars) - 1): #minus one bc u can make n-1 pairs from n letters: IE: in apple for example u can make ap, pp, pl, and le which is four pairs which is 5-1
            pair = (chars[i], chars[i+1]) #this is a tuple: learned abt this in codehs for coding solutions. even after all my years of coding i did not know what it was until this year...
            if pair in ps:
                ps[pair] += f
            else:
                ps[pair] = f


def merge(pair):
    global vocab
    nVcb = {}
    pr = ' '.join(pair)
    r = ''.join(pair)
    for w in vocab:
        nw = w.replace(pr, r) #basically what i just did here is that since all of the seperate characters are seperated by spaces, by deleting the space, u end up with one token for those characters: effectively merging them
        nVcb[nw] = vocab[w]

    vocab = nVcb


def translate(merges):
    # words = []
    # for w in raw_text.split():
    #     cTokens = list(w) + ['</w>']
    #     for p in merges:
    #         i = 0
    #         while (i < len(cTokens) - 1):
    #             if cTokens[i] == p[0] and cTokens[i + 1] == p[1]:
    #                 cTokens = cTokens[:i] + [''.join(p)] + cTokens[i+2:] #ok so basically we are checking if we are looking at a known merge, then u will delete those two tokens and add in one token that holds their pair
    #             else:
    #                 i+=1
    #     words.extend(cTokens)
    # return words
    ranks = {p : i for i, p in enumerate(merges)}
    words = []
    for w in raw_text.split():
        cTokens = list(w) + ['</w>']
        while True:
            toMerge = None
            bestR = float('inf')
            mergeI = -1
            for i in range(len(cTokens) - 1):
                p = (cTokens[i], cTokens[i+1])
                r = ranks.get(p, float('inf'))
                if r < bestR:
                    bestR = r
                    toMerge = p
                    mergeI = i
            if toMerge is None:
                break
            cTokens = cTokens[:mergeI] + [''.join(toMerge)] + cTokens[mergeI+2:]
        words.extend(cTokens)
    return words




merges = []

chars = set()
for w in vocab:
    for c in w.split():
        chars.add(c)

vocSize = len(chars)

while vocSize < 50000:
    pairs()
    if not ps:
        break

    mostFreq = None
    hiFrq = 0
    for p,f in ps.items():
        if f > hiFrq:
            hiFrq = f
            mostFreq = p
    
    merge(mostFreq)
    merges.append(mostFreq)
    vocSize += 1
    if(vocSize % 50 == 0):
        bar = '█' * int(vocSize / 50000 * 20)
        percent = (vocSize + 1) / 50000 * 100

        sys.stdout.write(f'\rProgress: |{bar:<20}| {percent:.1f}% |')
        sys.stdout.flush()


dim = 1024
tokens = translate(merges)

bpeVoc = {}
for t in tokens:
    if t not in bpeVoc:
        bpeVoc[t] = np.random.uniform(-1/math.sqrt(dim),1/math.sqrt(dim), size=dim)

bpeVoc["<PAD>"] = np.zeros(dim)
np.save('vocab.npy', bpeVoc)
print("saved the voc file")

with open('merges.txt', 'w', encoding='utf-8') as f:
    for pair in merges:
        f.write(pair[0] + ' ' + pair[1] + '\n')


runpy.run_path("myai.py")
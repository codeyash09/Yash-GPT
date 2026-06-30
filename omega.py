import cupy as n
import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
# import seaborn as sns
from pulse import HeatmapCreatorBG


#globals


if __name__ == '__main__':
    aConfig = [2,0,0,1,1]

    vis = HeatmapCreatorBG(displayConfig=aConfig, file="heatmaps.pdf")


    new = False
    dictLock = False
    train = True
    dim = 1024
    vocabSize = 0
    num_heads = 16
    bestGloss = 15
    trust = 1
    trust_lr = 0.05
    trustEngaged = False
    lastLoss = bestGloss
    trustMobile = False

    accusteps = 128
    wDecay = 0.001



    #temp
    #tepm

    num_layers = 24
    learning_rate = 0.001/math.sqrt(num_layers) #the lr is rly high but I did this so it will cause the model to jump out of plateaus. Nvm im just make it warm up to the correct size

    # base_lr = 0.001/math.sqrt(num_layers)
    base_lr = 0.0002
    batch = 1 #batch size cuz i did not want to wait like three weeks for it to hit 4 GLoss. MAX MY GPU CAN HANDLE: I TRIED 8 but it would offload to the cpu


    GLoss = 0 #to be deleted later

    Counterx = 0 #to be deleted later



    used = []
    for i in range(num_heads):
        used.append(0)

    curve = 1.0


    xs = []
    ys = []


    #globals ^


    glosses = []



    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    trust = n.load('trust.npy')[0] if os.path.exists('trust.npy') else trust
    #utils

    def normLayer(x, smallConst): #https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        mean = n.mean(x, keepdims=True, axis=-1)
        variance = n.var(x, keepdims=True, axis=-1)
        variance = n.maximum(variance, 1e-5)

        return ((x-mean)/n.sqrt(variance + smallConst)) #math.sqrt did not work

    def backNormLayer(out, x, smallConst): # this function came from a patchwork of articles https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/, https://veitner.bearblog.dev/backprob-through-layernorm/,
        # m = n.mean(out, keepdims=True, axis=-1)

        # v = n.var(x, keepdims=True, axis=-1)

        # v = n.maximum(v, 1e-5)
        
        # x_norm = normLayer(x, smallConst)

        # dx = 1/(n.sqrt(v+smallConst)) * (out - m - x_norm * n.mean(x_norm * out, keepdims=True, axis=-1))

        # return dx

        N = dim
        m = n.mean(x, keepdims=True,axis=-1)
        v = n.var(x, keepdims=True, axis=-1)
        stdn1 = 1.0/n.sqrt(v + smallConst)

        xn = (x-m) * stdn1

        dx = (1.0/N) * stdn1 * (N * out - n.sum(out, axis=-1, keepdims=True) - xn * n.sum(out * xn, keepdims=True, axis=-1))
        return dx


    def xinit(rows, cols, scale=1.0): #https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
        # limit = math.sqrt(6/(rows+cols)) * (1/math.sqrt(2*num_layers))
        limit = math.sqrt(2.0 / rows) * scale #He init now bc i want to speed up my start
        return n.random.uniform(-limit, limit, size=(rows, cols))


    def softmax(v): 
        exp_vector = n.exp(v - n.max(v, axis=-1, keepdims=True))
        probabilities = exp_vector / n.sum(exp_vector, axis=-1, keepdims=True)
        return probabilities
        

    def gelu(x):
        return x * 0.5 * (1.0 + n.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * n.power(x, 3))))

    def back_gelu(x):
        sech2 = lambda u: 1.0 - n.tanh(u)**2
        const1 = math.sqrt(2.0 / math.pi)
        inner = const1 * (x + 0.044715 * n.power(x, 3))
        return 0.5 * (1.0 + n.tanh(inner)) + 0.5 * x * sech2(inner) * const1 * (1.0 + 3 * 0.044715 * n.power(x, 2))


    def translate(word, merges): #literally the same as the other translate func from bpe.py
        
        cTokens = list(word) + ['</w>']
        for p in merges:
            i = 0
            while (i < len(cTokens) - 1):
                if cTokens[i] == p[0] and cTokens[i + 1] == p[1]:
                    cTokens = cTokens[:i] + [''.join(p)] + cTokens[i+2:]
                else:
                    i+=1
        return cTokens

    with open('input.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    merges = []
    with open('merges.txt', 'r', encoding='utf-8') as f:
        for l in f:
            a, b = l.strip().split(' ', 1)
            merges.append((a,b))


    cleaned = raw_text.replace('-', " - ").replace('.', " . ").replace(',', " , ").replace('?', " ? ").replace('!', " ! ").replace(':', " : ").replace(';', " ; ").replace('--', " -- ").replace("'", " ' ").replace('"', ' " ').replace('(', " ( ").replace(')', " ) ").replace('[', " [ ").replace(']', " ] ").replace('—'," — ").replace('”', " ” ").replace('–', ' – ').replace(' s ', ' s ').replace('“', ' “ ').lower().split()
    wordCount = {}
    for word in cleaned:
        if not word in wordCount:
            wordCount[word] = 0
        wordCount[word] +=1
        

    word_weights = {}
    vocabSize = len(cleaned)




    for word, count in wordCount.items():
        word_weights[word] = math.log(vocabSize / count) + 1.0
        # word_weights[word] = 1.0



    #dictionary and vocab loading

    dict = {}

    if not new:
        if os.path.exists('vocab.npy'):
            dict = np.load('vocab.npy', allow_pickle=True).item()
            dict = {word: n.array(vector) for word, vector in dict.items()}
        else:
            new = True



    if(new or len(dict) == 0):

        dict = {}

        def create_dict(txt):
            for word in cleaned:
                if(not word in dict):
                    dict[word] = n.array(n.random.uniform(-1/math.sqrt(dim),1/math.sqrt(dim), size=dim))

        create_dict(cleaned)

        np.save('vocab.npy', dict)

    if(new or len(dict) == 0):
        dict['<PAD>'] = n.zeros(dim)
        np.save('vocab.npy', n.asnumpy(dict))



    words = list(dict.keys())
    vectors = n.stack([dict[word] for word in words])


    dictionaryLookup = {}


    for index, word in enumerate(words):
        dictionaryLookup[word] = index



    #dictionary and vocab loading ^


    position = n.arange(dim)[:, n.newaxis]
    dimension = n.arange(dim)[n.newaxis, :]
    rates = 1 / n.power(10000, (2 * (dimension // 2)) / n.float32(dim))
    PE = n.zeros((dim, dim))
    PE[:, 0::2] = n.sin(position * rates[:, 0::2])
    PE[:, 1::2] = n.cos(position * rates[:, 1::2])



    ww = n.zeros(len(words)) #added this bc looking up for weights inside the loop took a lot of time: basically made a lookup table


    if os.path.exists('ww.npy'):
        ww = n.load('ww.npy')
    else:
        q = 0
        for word, val in word_weights.items():
            tList = translate(word, merges)
            if q % 50 == 0:
                bar = '█' * int(q / len(word_weights) * 20)
                percent = (q + 1) / len(word_weights) * 100

                sys.stdout.write(f'\rWW Progress: |{bar:<20}| {percent:.1f}% |')
                sys.stdout.flush()
            q+=1
            for tok in tList:
                if tok in dictionaryLookup:
                    ww[dictionaryLookup[tok]] = val
    #utils ^

        n.save('ww.npy', ww)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #training STAGE 1: CONTEXTUALIZE WORD MATRIX

    # def AttentionHead(segment, next_word, Wk, Wo, Wq, Wv, Wh1, Wh2): #segment length = dim
    def AttentionHead(segments, targets, Wk, Wq, Wv, Wh1, Wh2, Wo_final, mWk, mWv, mWq, mWh2, mWh1, vWk, vWv, vWq, vWh2, vWh1,t, b1, b2, ep): #segment length = dim
        global glosses

        global vectors 
        cont = True
        output = ""
        if(segments.shape[1] != int(dim)):
            cont = False

        if(cont):
            for i in range(len(used)):
                if(used[i] == 0):
                    id = i
                    break

            bSize = segments.shape[0]

            # if os.path.exists('Wo_' + str(id) + '.npy') and os.path.getsize('Wo_' + str(id) + '.npy') > 0:
            #     Wq = np.load(f'Wq_{id}.npy')
            #     Wk = np.load(f'Wk_{id}.npy')
            

            #     Wv = np.load(f'Wv_{id}.npy')
            #     Wo = np.load(f'Wo_{id}.npy')
            # else:
            #     Wq = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(dim, int(dim / num_heads)))
            #     Wk = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(dim, int(dim / num_heads)))
            #     Wv = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(dim, int(dim / num_heads)))
            #     Wo = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(int(dim / num_heads), len(dict.keys())))

            
            # X = n.array([dict[y] for y in segment], copy=True) Apparently this is slow for cupy: bc it goes one by one, now i have to fix...
            # indexes = n.array([dictionaryLookup[word] for word in segment])

            # X = vectors[indexes].copy() + PE #when i did not copy it, it would change the actual vectors when i used the actual dict but now im still doing it even tho it may not be necessary
            

            
            # for i in range(len(X)):

            #     for j in range(dim):

            #         if(j % 2 == 0):

            #             X[i][j] += math.sin(i / (10000**((2*j) / dim)))

            #         else:

            #             X[i][j] += math.cos(i / (10000**((2*j) / dim)))


        


            #Previous Single Head Attention Trial
            # Q = X @ Wq
            # K = X @ Wk
            # V = X @ Wv

            # length = X.shape[0] #size of dim1
            # mask = n.tril(n.ones((length, length))) #take the lower triangle made up of ones of size base dim1 and height dim1

            # scores = (Q @ K.T) / math.sqrt(dim)

            # scores = n.where(mask == 0, -1e9, scores)

            # A =softmax(scores)
            
            # Z = A @ V



            # Z = normLayer(Z + X, 1e-5) #residual connection: dekut-dsail.github.io/tutorials/transformer-architecture/6.%20Layer%20Normalisation%20&%20Residual%20Connection.html

            # output = Z[-1] @ Wo
            #Previous Single Head Attention Trial ^

            #Attempting MultiHead



















            # lengthOfSegment = X.shape[0] #rows of X aka # of words
            lengthOfSegment = segments.shape[1]
            
            dimPerHead = int(dim / num_heads)

            # Q = X @ Wq
            # K = X @ Wk
            # V = X @ Wv

            # Q = Q.reshape(lengthOfSegment, num_heads, dimPerHead) #basically we are splitting it into four arrays
            # V = V.reshape(lengthOfSegment, num_heads, dimPerHead)
            # K = K.reshape(lengthOfSegment, num_heads, dimPerHead)

            # Q = n.transpose(Q, (1,0,2)) #swap the first two dimensions Heads go first and then segment length, we do this bc after the Q = X @ Wq, our shape is (segment, dim) so even after our reshape, we wont have heads first which is y we must swap
            # K = n.transpose(K, (1,0,2))
            # V = n.transpose(V, (1,0,2))

            # scores = (Q @ n.transpose(K, (0,2,1))) / math.sqrt(dimPerHead) #swap the last two to do .T for the K. basically we are following the attenion mechanism formula


            # mask = n.tril(n.ones((lengthOfSegment, lengthOfSegment))) #same as last time: its to hide the future words in the weights
            # scores = n.where(mask == 0, -1e9, scores)

            # A = softmax(scores)

            # Z = A @ V #u get the shape (heds, length, dim) but we will now need to go back and undo this 

            # Z = n.transpose(Z, (1,0,2)) #swap the first two dims to get len first

            # Z = Z.reshape(lengthOfSegment, dim) #combine all of the heads to get one large vector


            
            # Z = Z + X

            # Zb = Z
            # Z = normLayer(Z, 1e-5)

            # hidden1 = Z @ Wh1 #got rid of Z[-1] bc this way it trains more than it used to by training off of all of the words



            # hidden1 = n.maximum(0, hidden1)
            # global Counterx
            # if(Counterx % 1000 == 0):
            #     active_neurons = n.sum(hidden1 > 0) / hidden1.size
            #     print(f" | ReLU Activity: {active_neurons:.2%}")
            # Counterx+=1

            # hidden2 = hidden1 @ Wh2

            # hidden2 += Z
            # h2b = hidden2
            # hidden2 = normLayer(hidden2, 1e-5)

            # output = hidden2 @ Wo

            all_X = []
            all_X_pm = []
            all_A = []
            all_V = []
            all_Q = []
            all_K = []
            all_h1 = []
            all_Zb = []
            all_h2b = []

            currentX = (vectors[segments].copy()) + PE
            mask = n.tril(n.ones((lengthOfSegment, lengthOfSegment))) 

            for i in range(num_layers):
                X = normLayer(currentX, 1e-5)
                all_X.append(X.copy())

                Q = X @ Wq[i]
                K = X @ Wk[i]
                V = X @ Wv[i]

                Q = Q.reshape(bSize, lengthOfSegment, num_heads, dimPerHead) 
                V = V.reshape(bSize, lengthOfSegment, num_heads, dimPerHead)
            
                K = K.reshape(bSize, lengthOfSegment, num_heads, dimPerHead)

                # Q = n.transpose(Q, (1,0,2)) 
                # K = n.transpose(K, (1,0,2))
                # V = n.transpose(V, (1,0,2))

                Q = n.transpose(Q, (0,2,1,3)) 
                V = n.transpose(V, (0,2,1,3))
                K = n.transpose(K, (0,2,1,3))

                scores = (Q @ n.transpose(K, (0,1,3,2))) / math.sqrt(dimPerHead) 


                scores = n.where(mask == 0, -1e9, scores)

                A = softmax(scores)


                all_A.append(A.copy())
                all_V.append(V.copy())
                all_Q.append(Q.copy())
                all_K.append(K.copy())


                Z = A @ V 

                Z = n.transpose(Z, (0,2,1,3)) 

                Z = Z.reshape(bSize, lengthOfSegment, dim)

                currentX = currentX + (Z / math.sqrt(num_layers))
                # global curve
                # Z = (Z / math.sqrt(num_layers)) * curve + currentX

                # Zn = normLayer(Z, 1e-5)


                all_Zb.append(Z.copy())



                X = normLayer(currentX, 1e-5)
                all_X_pm.append(X.copy())

                hidden1 = X @ Wh1[i] 

                # hidden1 = n.maximum(hidden1 * 0.01, hidden1) #LEAKY RELU: THE NEW AMAZING FIX to my "ubiquitous" dying nueron problem: sry but i learned too many new words for the sat and i had to use it.

                hidden1 = gelu(hidden1)

                global Counterx
                if Counterx % ((num_layers*1000) + 1) == 0:
                    
                    active_ratio = n.mean(hidden1 > 0) 
                    print(f" | Layer {i} ReLU Activity: {active_ratio:.2%} | Weight Scale: {n.linalg.norm(Wh1[i])}")
                Counterx+=1

                all_h1.append(hidden1)

                hidden2 = hidden1 @ Wh2[i]

                all_h2b.append(hidden2.copy())

                h2b = hidden2
            
                currentX = currentX + hidden2


                

            output = currentX @ Wo_final




            
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------Forward Pass Above------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            #Attempting MultiHead ^


            Probabilities = softmax(output)

            if(Counterx % (num_layers*10) == 0):
                cpu_A = [n.asnumpy(A_layer) if hasattr(A_layer, 'get') else A_layer for A_layer in all_A]
                vis.log_matrix(np.stack(cpu_A, axis=0))


            # i hit a probabilities are not non-negative error: so fix is below: DO NOT MAKE THIS MISTAKE AGAIN-> I had to restart which ruined an hour's worth of training
            # Probabilities = n.clip(Probabilities, 1e-9, 1.0)
            # Probabilities /= n.sum(Probabilities, axis=-1, keepdims=True) 




            weights = ww[targets][:, n.newaxis] #finding weights for each word to encourage learning rarer words

            correct = Probabilities[n.arange(bSize)[:, n.newaxis], n.arange(lengthOfSegment), targets] #basically this is the loss stuff its cool but only for visuals and this is the loss for the entire segment
    
            Loss = -n.mean(n.log(correct + 1e-10)) #loss for entire segment

            # if i % 100 == 0:
            #     print(f"\n=== Step {i} ===")
            #     print(f"Loss: {Loss:.4f}")
            #     for layer_idx in range(num_layers):
            #         print(f"Layer {layer_idx}: Wq_norm={n.linalg.norm(grad_Wq[layer_idx]):.4f}, "
            #             f"Wh1_norm={n.linalg.norm(grad_Wh1[layer_idx]):.4f}")
            #     print(f"Output range: [{n.min(output):.2f}, {n.max(output):.2f}]")
            #     print(f"Embedding norm: {n.mean(n.linalg.norm(vectors, axis=1)):.4f}")
            Error = Probabilities.copy()    

            Error[n.arange(bSize)[:, n.newaxis], n.arange(lengthOfSegment), targets] -= 1.0

            Error /= (bSize * lengthOfSegment)
            Error *= weights.reshape(bSize, lengthOfSegment, 1)

            dE = Error @ Wo_final.T

            grad_Wo_final = n.sum(n.transpose(currentX, (0,2,1)) @ Error, axis=0) / bSize



            grad_Wq, grad_Wv, grad_Wk, grad_Wh1, grad_Wh2= [None] * num_layers, [None] * num_layers, [None] * num_layers, [None] * num_layers, [None] * num_layers
            for i in reversed(range(num_layers)):
                
                dx_res = dE.copy()
                dhidden2 = dE
                
                # grad_Wh2[i] = n.sum(n.transpose(all_h1[i], (0,2,1)) @ dhidden2, axis=0) / (bSize *lengthOfSegment)
                grad_Wh2[i] = n.sum(n.transpose(all_h1[i], (0,2,1)) @ dhidden2, axis=0) / bSize
                dhidden1 = dhidden2 @ Wh2[i].T

                dhidden1 = dhidden1 * back_gelu(all_h1[i]) #derivative of gelu

                # grad_Wh1[i] = n.sum(n.transpose(normLayer(all_Zb[i], 1e-5), (0,2,1)) @ dhidden1, axis=0) / (bSize *lengthOfSegment) #New, u need this bc now the input is different from the normalized input while before it was not
                grad_Wh1[i] = n.sum(n.transpose(all_X_pm[i], (0,2,1)) @ dhidden1, axis=0) / bSize
                
                dX_pm = dhidden1 @ Wh1[i].T

                dx_out = backNormLayer(dX_pm, all_X_pm[i], 1e-5)
                dattn = dx_out + dx_res

                dattn_res = dattn.copy()
                


                dz = dattn / math.sqrt(num_layers)
                dz_heads = dz.reshape(bSize, lengthOfSegment, num_heads, dimPerHead)
                dz_heads = n.transpose(dz_heads, (0,2,1,3)) 


                dA = dz_heads @ n.transpose(all_V[i],(0,1,3,2)) 
                dV = n.transpose(all_A[i], (0,1,3,2)) @ dz_heads


                dSoftmax = all_A[i] * (dA - n.sum(dA * all_A[i], axis=-1, keepdims=True))
                dSoftmax /= math.sqrt(dimPerHead)
                dSoftmax = n.where(mask == 0, 0, dSoftmax)

                dQ = dSoftmax @ all_K[i]
                dK = n.transpose(dSoftmax, (0,1,3,2)) @ all_Q[i]

                dQ = n.transpose(dQ, (0,2,1,3)) 
                dQ = n.ascontiguousarray(dQ).reshape(bSize, lengthOfSegment, dim)

                dK = n.transpose(dK, (0,2,1,3)) 
                dK = n.ascontiguousarray(dK).reshape(bSize, lengthOfSegment, dim)

                dV = n.transpose(dV, (0,2,1,3)) 
                dV = n.ascontiguousarray(dV).reshape(bSize, lengthOfSegment, dim)


                
                grad_Wv[i] = n.sum(n.transpose(all_X[i], (0,2,1)) @ dV, axis=0) / bSize
                grad_Wk[i] = n.sum(n.transpose(all_X[i], (0,2,1)) @ dK, axis=0) / bSize
                grad_Wq[i] = n.sum(n.transpose(all_X[i], (0,2,1)) @ dQ, axis=0) / bSize

                
                Sx = dQ @ Wq[i].T + dK @ Wk[i].T + dV @ Wv[i].T
                # Sx = Sx / math.sqrt(num_layers) * curve
                # rError = Sx + dhidden2
                dx_in = backNormLayer(Sx, all_X[i], 1e-5)
                dE = dE + dx_in
                

            

                # for i, word in enumerate(segment):
                #     vectors[word] -= (learning_rate * 0.01) * Sx[i]
                #     normalize = n.linalg.norm(dict[word])
                #     if normalize > 0:
                #         vectors[word] /= normalize
            

            t+=1
            # for i in range(num_layers):
                # grads = [grad_Wv[i], grad_Wk[i], grad_Wq[i], grad_Wh2[i], grad_Wh1[i]]
                # weights = [Wv, Wk, Wq, Wh2, Wh1]

                # for idx in range(len(grads)):
                #     g = grads[idx]
                #     if g is None:
                #         continue
                    
                    

                #     norm = n.linalg.norm(g)
                    
                #     if norm > 1.0:
                #         grads[idx]*=1.0/norm
                #     weights[idx][i] -= learning_rate * grads[idx]
                #ADAM
                # g_norm_i = n.linalg.norm(grad_Wh1[i])
                # if g_norm_i > 1000: 
                #     print(f"Layer {i} is exploding! Norm: {g_norm_i}")

                # g_norm = n.sqrt(sum(n.sum(g**2) for g in [grad_Wv[i], grad_Wk[i], grad_Wq[i], grad_Wh2[i], grad_Wh1[i], grad_Wo_final] if g is not None))
                # glob_scl = min(1.0, 1.0 / (g_norm + 1e-8))

                # # if g_norm > 1.0:
                # #     for g in [grad_Wv[i], grad_Wk[i], grad_Wq[i], grad_Wh2[i], grad_Wh1[i]]:
                # #         if g is not None:
                # #             g *= 1.0 / g_norm

                # for g, m, v, w in zip(
                #     [grad_Wv[i], grad_Wk[i], grad_Wq[i], grad_Wh2[i], grad_Wh1[i]],
                #     [mWv[i], mWk[i], mWq[i], mWh2[i], mWh1[i]],
                #     [vWv[i], vWk[i], vWq[i], vWh2[i], vWh1[i]],
                #     [Wv[i], Wk[i], Wq[i], Wh2[i], Wh1[i]],
                # ):
                #     if g is None:
                #         continue
                    
                #     g_norm_i = n.linalg.norm(g)
                #     if g_norm_i > 1.0:
                #         g = g * (1.0 / g_norm_i)
                    
                #     g = g * glob_scl

                
                #     m[:] = b1 * m + (1 - b1) * g
                #     v[:] = b2 * v + (1 - b2) * (g*g)

                #     mCorrected = (m / (1 - b1**t))
                #     vCorrected = (v / (1 - b2**t))

                #     u = mCorrected / (n.sqrt(vCorrected + ep))

                


                    # w -= u * learning_rate
                    # if(trustEngaged):
                    #     threshold = n.maximum(n.abs(w)*trust, 1e-3)
                    #     scaled = n.clip(u, -threshold, threshold)
                    #     w -= scaled * learning_rate
                    # else:
                    #     w -= u * learning_rate




            
                #active_ratio = n.mean(hidden1 > 0) 

                #new training system: i call it adaptive alternating decay

                # distAR = (active_ratio - 0.5) / 1
                # distSize = -(n.linalg.norm(Wh1[i]) - 10) / 5
                #active_ratio_fctr = 1.0 + (0.5 - active_ratio) * learning_rate
                #size_fctr = 1.0 + (10 - n.linalg.norm(Wh1[i])) * (learning_rate)
                # if n.linalg.norm(Wh1[i]) > 10 and active_ratio > 0.50 and i%2 == 1:
                #     Wh1[i] *= 0.92
                #     Wh2[i] *= 0.92
                # if i% 2 == 1:
                    

                #     diff = (0.5 - active_ratio)
                #     norm = n.linalg.norm(Wh1[i])
                #     sizeDiff = 10-norm


                #     # vector = Wh1[i] / (norm + 1e-9)

                
                #     # moveAmt = (vector * diff * learning_rate * 0.1) + (vector * sizeDiff * learning_rate * 0.1)

                #     # Wh1[i] += moveAmt

                #     # Wh2[i] += (Wh2[i] / n.linalg.norm(Wh2[i] + 1e-9)) * (diff + sizeDiff) * (learning_rate * 0.1)

                #     Wh1[i] *= 1.0 + (diff + sizeDiff) * learning_rate * 0.1

                #     norm2 = n.linalg.norm(Wh2[i])
                #     Wh2[i] *= 1.0 + (diff + sizeDiff) * learning_rate * 0.1
                    # if distAR > distSize:
                    #     #🪦 0.92 rests here... its a sad day for all of us: we gather here to mourn the death of a loyal weight decaying constant: 0.92. he always  rose to the challenge but then we became too greedy and he kinda started doing bad things to the weight scales. 2/10/26 - Remeber though 0.92 will never die because it will live on in our hearts.
                    #     Wh1[i] *= (1 - learning_rate) * distAR
                    #     Wh2[i] *= (1 - learning_rate) * distAR 
                    # else:
                    #     Wh1[i] *= (1 + learning_rate) * distSize
                    #     Wh2[i] *= (1 + learning_rate) * distSize
                    
            
            # Wo_final -= learning_rate * grad_Wo_final
            # nrm = n.linalg.norm(grad_Wo_final)
            # if nrm > 1.0:
            #     grad_Wo_final*=1.0/nrm
            # mWo[:] = b1 * mWo + (1 - b1) * grad_Wo_final
            # vWo[:] = b2 * vWo + (1 - b2) * (grad_Wo_final*grad_Wo_final)

            # mCorrected = (mWo / (1 - b1**t))
            # vCorrected = (vWo / (1 - b2**t))

            # u = mCorrected / (n.sqrt(vCorrected + ep))

            


            # # Wo_final -= u * learning_rate

            if(not dictLock):
                v_grad = n.clip(dE, -1.00, 1.0)

                for batch in n.unique(segments.flatten()):
                    
                    vectors[batch] -= (learning_rate * n.sum(v_grad[(segments == batch)], axis=0))


                #DUDE.... I trained for 24 whole hours on my gpu on 1/20/26 without this block and it caused the words to explode...
                nrm = n.linalg.norm(vectors, axis=1, keepdims = True)
            
                vectors = n.where(nrm > 1.0, vectors/nrm, vectors) # deleted the max scaling bc i realized if everything is becoming huge then there is no point of it

            used[id] = 0

            # GLoss += Loss
            if(len(glosses) < 100):
                glosses.append(float(Loss.get() if hasattr(Loss, 'get') else Loss))
            else:
                glosses.pop(0)
                glosses.append(float(Loss.get() if hasattr(Loss, 'get') else Loss))

            global GLoss
            GLoss += float(Loss.get() if hasattr(Loss, 'get') else Loss)
            return Wk, Wq, Wv, Wh1, Wh2, Wo_final, mWk, mWv, mWq, mWh2, mWh1, vWk, vWv, vWq, vWh2, vWh1,t, b1, b2, ep, grad_Wv, grad_Wk, grad_Wq, grad_Wh2, grad_Wh1, grad_Wo_final


                    


                    

            # grad_Wo = (hidden2.T @ Error) / lengthOfSegment

            # dhidden2 = Error @ Wo.T

            # dhidden2 = backNormLayer(dhidden2, h2b, 1e-5)

            # grad_Wh2 = (hidden1.T @ dhidden2) / lengthOfSegment

            # dhidden1 = dhidden2 @ Wh2.T

            # dhidden1[hidden1 <= 0] = 0

            # grad_Wh1 = (Z.T @ dhidden1) / lengthOfSegment


        
            # LastError = dhidden1 @ Wh1.T

            # dz = dhidden2

            # dz = backNormLayer(dz, Zb, 1e-5)
            

            # dz_heads = dz.reshape(lengthOfSegment, num_heads, dimPerHead) #basically doing the same thing as before but now we are doing it by splitting the backprop into multi heads
            # dz_heads = n.transpose(dz_heads, (1,0,2)) #literally the same as before, make heads first -> we will switch this back to len first at the end


            # dA = dz_heads @ n.transpose(V, (0,2,1)) #basically when doing multihead remember that .T becomes transpose(..., (0,2,1))
            # dV = n.transpose(A, (0,2,1)) @dz_heads # remember how Z = A @ V, We are undoing that by moving V to the other side and tranposing A and then mult by the d/dx of z per head
            # dSoftmax = A * (dA - n.T(dA * A, axis=-1, keepdims=True))
            # dSoftmax /= math.sqrt(dimPerHead)
            # dSoftmax = n.where(mask == 0, 0, dSoftmax)

            # # dA = dz @ V.T
            # # dSoftmax = A * (dA - n.sum(dA * A, axis=1, keepdims=True))
            # # dSoftmax /= math.sqrt(dim) --------------------------------------> Old stuff from single backprop

            # dQ = dSoftmax @ K
            # dK = n.transpose(dSoftmax, (0,2,1)) @ Q #so this stuff is basically the stuff below, but we need to do it in multiple steps since this has multiple heads and tranposing is not as simple anymore

            # dQ = n.transpose(dQ, (1,0,2)) #we are back to how it was length first
            # dK = n.transpose(dK, (1,0,2)) 
            # dV = n.transpose(dV, (1,0,2)) 

            # dQ = dQ.reshape(lengthOfSegment, dim) #backprop is not that bad...its just following the same stuff we did to get the output but in reverse: i just realized that we have already done this
            # dK = dK.reshape(lengthOfSegment, dim)
            # dV = dV.reshape(lengthOfSegment, dim)


            # # grad_Wv = X.T @ (A.T @ dz)
            # # grad_Wk = X.T @ (dSoftmax.T @ (X @ Wq))
            # # grad_Wq = X.T @ (dSoftmax @ (X @ Wk))
            # grad_Wv = X.T @ dV #we are following the same thing basically
            # grad_Wk = X.T @ dK
            # grad_Wq = X.T @ dQ 
        
            # Sx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
            # # Sx[-1] += LastError @ Wv.T



        
            # for g in [grad_Wo, grad_Wv, grad_Wk, grad_Wq, Sx, grad_Wh2, grad_Wh1]:
            #     norm = n.linalg.norm(g)
            #     if norm > 0.5:
            #         g *= (0.5 / norm) # normalize and clip
            #     g+= n.random.normal(0,0.001*base_lr, g.shape) #noise to help out since it would just get such on one of them: i think the term is temperature but im not sure

        
            # Wo -= learning_rate * grad_Wo
            # Wv -= learning_rate * grad_Wv
            # Wk -= learning_rate * grad_Wk
            # Wq -= learning_rate * grad_Wq
            # Wh1 -= learning_rate * grad_Wh1
            # Wh2 -= learning_rate * grad_Wh2

            # # for i, word in enumerate(segment):
            # #     vectors[word] -= (learning_rate * 0.01) * Sx[i]
            # #     normalize = n.linalg.norm(dict[word])
            # #     if normalize > 0:
            # #         vectors[word] /= normalize
            # if(not dictLock):
            #     vectors[indexes] -= (learning_rate * 0.001) * Sx
                
                
        
            # used[id] = 0

            

            # # np.save(f'Wk_{id}.npy', Wk)
            # # np.save(f'Wo_{id}.npy', Wo)
            # # np.save(f'Wq_{id}.npy', Wq)
            # # np.save(f'Wv_{id}.npy', Wv)

            
            # global GLoss
            # GLoss += Loss
            # Wh1 *= 0.92 #it seems aggressive but i had to bc my relu went all the way up to 70.21% and my GLoss jumped to 6. Im now back on the grind to get it to 4.29. also this may seem like im killing it but the numbers dont lie. I pinned ReLU to 50%
            # Wh2 *= 0.92
            
            # return Wk, Wo, Wq, Wv, Wh1, Wh2

        
    # for id in used:
    #     if(id == 0):
    #         if os.path.exists('Wo_' + str(id) + '.npy') and os.path.getsize('Wo_' + str(id) + '.npy') > 0:
    #             Wq = n.asarray(np.load(f'Wq_{id}.npy'))
    #             Wk = n.asarray(np.load(f'Wk_{id}.npy'))
    #             Wv = n.asarray(np.load(f'Wv_{id}.npy'))
    #             Wo = n.asarray(np.load(f'Wo_{id}.npy'))
    #             Wh1 = n.asarray(np.load(f'Wh1_{id}.npy'))
    #             Wh2 = n.asarray(np.load(f'Wh2_{id}.npy'))
    #         else:
    #             Wq = xinit(dim, int(dim))
    #             Wk = xinit(dim, int(dim))
    #             Wv = xinit(dim, int(dim))
    #             Wo = xinit(int(dim), len(dict.keys()))
    #             Wh1 = xinit(dim, 4 * dim)
    #             Wh2 = xinit(4* dim, dim)
                
            
    #         break


    layer_files = ['Wq_layers.npy', 'Wk_layers.npy', 'Wv_layers.npy', 
                'Wo_final.npy', 'Wh1_layers.npy', 'Wh2_layers.npy']


    if all(os.path.exists(f) for f in layer_files):
        Wq_raw = np.load('Wq_layers.npy', allow_pickle=True)
        Wk_raw = np.load('Wk_layers.npy', allow_pickle=True)
        Wv_raw = np.load('Wv_layers.npy', allow_pickle=True)
    
        Wh1_raw = np.load('Wh1_layers.npy', allow_pickle=True)
        Wh2_raw = np.load('Wh2_layers.npy', allow_pickle=True)
        Wo_final = n.asarray(np.load("Wo_final.npy"))

        Wq = [n.asarray(layer) for layer in Wq_raw]
        Wk = [n.asarray(layer) for layer in Wk_raw]
        Wv = [n.asarray(layer) for layer in Wv_raw]

        Wh1 = [n.asarray(layer) for layer in Wh1_raw]
        Wh2 = [n.asarray(layer) for layer in Wh2_raw]

    else:
        Wq, Wk, Wv, Wh1, Wh2 = [], [], [], [], []
        for i in range(num_layers):
            Wq.append(xinit(dim, dim))
            Wk.append(xinit(dim, dim))
            Wv.append(xinit(dim, dim))
            Wh1.append(xinit(dim, 4 * dim))
            Wh2.append(xinit(4 * dim, dim, scale=(1.0/n.sqrt(2*num_layers))))

        
        Wo_final = xinit(dim,len(words))

    def init_grad_buffers(weights_list):
        return [n.zeros_like(w) for w in weights_list]

    acc_Wq = init_grad_buffers(Wq)
    acc_Wk = init_grad_buffers(Wk)
    acc_Wv = init_grad_buffers(Wv)
    acc_Wh1 = init_grad_buffers(Wh1)
    acc_Wh2 = init_grad_buffers(Wh2)
    acc_Wo = n.zeros_like(Wo_final)


    #means for adam: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/



    a_fls = ['mWq_layers.npy', 'mWk_layers.npy', 'mWv_layers.npy', 'mWh1_layers.npy', 'mWh2_layers.npy', 'mWo.npy', 'vWq_layers.npy', 'vWk_layers.npy', 'vWv_layers.npy', 'vWh1_layers.npy', 'vWh2_layers.npy', 'vWo.npy', 'adm_t.npy', 'vV.npy', 'mV.npy']

    if all(os.path.exists(f) for f in a_fls):
        mWq = [n.asarray(l) for l in np.load('mWq_layers.npy', allow_pickle=True)]
        mWk = [n.asarray(l) for l in np.load('mWk_layers.npy', allow_pickle=True)]
        mWv = [n.asarray(l) for l in np.load('mWv_layers.npy', allow_pickle=True)]
        mWh1 = [n.asarray(l) for l in np.load('mWh1_layers.npy', allow_pickle=True)]
        mWh2 = [n.asarray(l) for l in np.load('mWh2_layers.npy', allow_pickle=True)]
        mWo = n.asarray(np.load('mWo.npy'))

        vWq = [n.asarray(l) for l in np.load('vWq_layers.npy', allow_pickle=True)]
        vWk = [n.asarray(l) for l in np.load('vWk_layers.npy', allow_pickle=True)]
        vWv = [n.asarray(l) for l in np.load('vWv_layers.npy', allow_pickle=True)]
        vWh1 = [n.asarray(l) for l in np.load('vWh1_layers.npy', allow_pickle=True)]
        vWh2 = [n.asarray(l) for l in np.load('vWh2_layers.npy', allow_pickle=True)]
        vWo = n.asarray(np.load('vWo.npy'))

        t = int(np.load('adm_t.npy')[0])
    else:
        mWq = [n.zeros_like(w) for w in Wq]
        mWk = [n.zeros_like(w) for w in Wk]
        mWv = [n.zeros_like(w) for w in Wv]
        mWh1 = [n.zeros_like(w) for w in Wh1]
        mWh2 = [n.zeros_like(w) for w in Wh2]
        mWo = n.zeros_like(Wo_final)
            
        #variances 
        vWq = [n.zeros_like(w) for w in Wq]
        vWk = [n.zeros_like(w) for w in Wk]
        vWv = [n.zeros_like(w) for w in Wv]
        vWh1 = [n.zeros_like(w) for w in Wh1]
        vWh2 = [n.zeros_like(w) for w in Wh2]
        vWo = n.zeros_like(Wo_final)

        t = 1 #time

    #adam globals
    b1 = 0.9 #beta 1
    b2 = 0.999 #beta 2
    ep = 10**-8 #epsilon


    def save_layers(filename, layer_list):
        obj_arr = np.empty(len(layer_list), dtype=object)
        for idx, layer in enumerate(layer_list):
            obj_arr[idx] = layer.get() 
        np.save(filename, obj_arr)
    steps = 20000000


    #indxs = n.array([dictionaryLookup[word] for word in cleaned])
    tkns = []
    q = 0

    cache = {}


    if os.path.exists('tkns.npy'):
        tkns = np.load('tkns.npy', allow_pickle=True).tolist()
    else:
        for w in cleaned:
            if w not in cache:

                cache[w] = translate(w, merges)
            tkns.extend(cache[w])
            if q % 50 == 0:
                bar = '█' * int(q / len(cleaned) * 20)
                percent = (q + 1) / len(cleaned) * 100

                sys.stdout.write(f'\rTkns Progress: |{bar:<20}| {percent:.1f}% |')
                sys.stdout.flush()
            q+=1
        np.save('tkns.npy', np.array(tkns,dtype=object))
    indxs = np.array([dictionaryLookup[t] for t in tkns if t in dictionaryLookup])


    if(train):
        explosion_cooldown = 0

        for i in range(22500, steps):

            x = cleaned
            # start = n.random.randint(int(dim + 1), len(x) - int(dim + 1) - 1)
            start = n.random.randint(0, len(indxs) - dim - 1, size=batch) #find all starting points for the batch
            # start = int(start)
            

            # ind = []
            # targets = []

            # for s in start:
            #     i_s = int(s.item())
            #     segment = [dictionaryLookup[word] for word in cleaned[i_s :i_s +dim]]
            #     ind.append(n.array(segment))

            #     target = [dictionaryLookup[word] for word in cleaned[i_s +1:i_s +dim+1]]
            #     targets.append(n.array(target))

            # ind = n.stack(ind)
            # targets = n.stack(targets)

            idx_offsets = np.arange(dim)
            startC = np.array(start.get())
            ind = n.array(indxs[startC[:, np.newaxis] + idx_offsets])
            targets = n.array(indxs[startC[:, np.newaxis] + idx_offsets + 1])

            Wk, Wq, Wv, Wh1, Wh2, Wo_final, mWk, mWv, mWq, mWh2, mWh1, vWk, vWv, vWq, vWh2, vWh1,t, b1, b2, ep, grad_Wv, grad_Wk, grad_Wq, grad_Wh2, grad_Wh1, grad_Wo_final = AttentionHead(ind, targets, Wk, Wq, Wv, Wh1, Wh2, Wo_final, mWk, mWv, mWq, mWh2, mWh1, vWk, vWv, vWq, vWh2, vWh1,t, b1, b2, ep)
            
            for layer in range(num_layers):
                acc_Wq[layer] += grad_Wq[layer]
                acc_Wk[layer] += grad_Wk[layer]
                acc_Wv[layer] += grad_Wv[layer]
                acc_Wh1[layer] += grad_Wh1[layer]
                acc_Wh2[layer] += grad_Wh2[layer]
            
            acc_Wo += grad_Wo_final
            # acc_V += v_grad

            if (i + 1) % accusteps == 0:
                for layer in range(num_layers):
                    acc_Wq[layer] /= accusteps
                    acc_Wk[layer] /= accusteps
                    acc_Wv[layer] /= accusteps
                    acc_Wh1[layer] /= accusteps
                    acc_Wh2[layer] /= accusteps
                acc_Wo /= accusteps
                
            
                for li in range(num_layers):
                    g_norm = n.sqrt(sum(n.sum(g**2) for g in [acc_Wv[li], acc_Wk[li], acc_Wq[li], acc_Wh2[li], acc_Wh1[li], acc_Wo] if g is not None))
                    glob_scl = min(1.0,1.0/(g_norm + 1e-8))
                    for g, m, v, w in zip(
                        [acc_Wv[li], acc_Wk[li], acc_Wq[li], acc_Wh2[li], acc_Wh1[li]],
                        [mWv[li], mWk[li], mWq[li], mWh2[li], mWh1[li]],
                        [vWv[li], vWk[li], vWq[li], vWh2[li], vWh1[li]],
                        [Wv[li], Wk[li], Wq[li], Wh2[li], Wh1[li]],
                    ):
                        
                        if g is None:
                            continue

                    
                        g_norm_i = n.linalg.norm(g)
                        if(g_norm_i > 1.0):
                            g = g * (1.0 / g_norm_i)

                        g = g * glob_scl
                        
                    
                        m[:] = b1 * m + (1 - b1) * g
                        v[:] = b2 * v + (1 - b2) * (g**2)

                        mCorrected = (m / (1 - b1**t))
                        vCorrected = (v / (1 - b2**t))

                        u = mCorrected / (n.sqrt(vCorrected + ep))


                    

                        w -= (u+wDecay*w) * learning_rate
                        if(trustEngaged):
                            threshold = n.maximum(n.abs(w)*trust, 1e-3)
                            scaled = n.clip(u, -threshold, threshold)
                            w -= (scaled+wDecay*w) * learning_rate
                        else:
                            w -= u * learning_rate
                
                nrm = n.linalg.norm(acc_Wo)

                if(nrm > 1.0):
                    acc_Wo *= 1.0/nrm         
            
                mWo[:] = b1 * mWo + (1 - b1) * acc_Wo
                vWo[:] = b2 * vWo + (1 - b2) * (acc_Wo**2)

                mCorrected = (mWo / (1 - b1**t))
                vCorrected = (vWo / (1 - b2**t))

                u = mCorrected / (n.sqrt(vCorrected + ep))

                


                Wo_final -= (u+wDecay*Wo_final) * learning_rate

                
            
                for layer in range(num_layers):
                    acc_Wq[layer].fill(0)
                    acc_Wv[layer].fill(0)
                    acc_Wk[layer].fill(0)
                    acc_Wh1[layer].fill(0)
                    acc_Wh2[layer].fill(0)

                acc_Wo.fill(0)


            if i % 100 == 0:
                percent = (i + 1) / steps * 100
                bar = '█' * int(i / steps * 20)
                lastLoss = GLoss
                GLoss = sum(glosses) / len(glosses) if glosses else 0.0
                if(GLoss < lastLoss):
                    if(trustEngaged and trustMobile):
                        trust *= (1+ trust_lr * (lastLoss - GLoss) * 0.01)
                else:
                    if(trustEngaged and trustMobile):
                        trust *= (1/(1+trust_lr * (GLoss - lastLoss) * 0.05))
                if(i == 0):
                    lock = GLoss
                else:
                    lock = GLoss
                sys.stdout.write(f'\rProgress: |{bar:<20}| {percent:.1f}% | {lock} | {dictLock} | {trust:.4f}')
                sys.stdout.flush()

                xs.append(i)
            
                ys.append(float(GLoss))

                plt.clf()
                plt.plot(xs,ys,'b-')
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.title("The Amazing Loss Tracker")
                plt.grid(True)
                plt.savefig('loss_curve.png')
                plt.close()

                n.get_default_memory_pool().free_all_blocks()
                n.get_default_pinned_memory_pool().free_all_blocks()


            if i % (steps * 0.00005) == 0:

                if i >= 1000:
                    GLoss = sum(glosses) / len(glosses) if glosses else 0.0

                    if GLoss > bestGloss + 0.05 and all(os.path.exists(f) for f in layer_files):
                        print(f"\n[EXPLOSION] Step {i}, GLoss {GLoss:.4f} > best {bestGloss:.4f}. Reloading checkpoint...")
                        
                        Wq = [n.asarray(l) for l in np.load('Wq_layers.npy', allow_pickle=True)]
                        Wk = [n.asarray(l) for l in np.load('Wk_layers.npy', allow_pickle=True)]
                        Wv = [n.asarray(l) for l in np.load('Wv_layers.npy', allow_pickle=True)]
                        Wh1 = [n.asarray(l) for l in np.load('Wh1_layers.npy', allow_pickle=True)]
                        Wh2 = [n.asarray(l) for l in np.load('Wh2_layers.npy', allow_pickle=True)]
                        Wo_final = n.asarray(np.load('Wo_final.npy'))
                        
                        mWq = [n.asarray(l) for l in np.load('mWq_layers.npy', allow_pickle=True)]
                        mWk = [n.asarray(l) for l in np.load('mWk_layers.npy', allow_pickle=True)]
                        mWv = [n.asarray(l) for l in np.load('mWv_layers.npy', allow_pickle=True)]
                        mWh1 = [n.asarray(l) for l in np.load('mWh1_layers.npy', allow_pickle=True)]
                        mWh2 = [n.asarray(l) for l in np.load('mWh2_layers.npy', allow_pickle=True)]
                        mWo = n.asarray(np.load('mWo.npy'))
                        vWq = [n.asarray(l) for l in np.load('vWq_layers.npy', allow_pickle=True)]
                        vWk = [n.asarray(l) for l in np.load('vWk_layers.npy', allow_pickle=True)]
                        vWv = [n.asarray(l) for l in np.load('vWv_layers.npy', allow_pickle=True)]
                        vWh1 = [n.asarray(l) for l in np.load('vWh1_layers.npy', allow_pickle=True)]
                        vWh2 = [n.asarray(l) for l in np.load('vWh2_layers.npy', allow_pickle=True)]
                        vWo = n.asarray(np.load('vWo.npy'))
                        
                        cpu_dict = np.load('vocab.npy', allow_pickle=True).item()
                        vectors = n.array(n.stack([n.asarray(cpu_dict[w]) for w in words]))
                        
                        base_lr = max(base_lr * 0.5, min_lr * 2)
                        learning_rate = base_lr
                        glosses = []
                        explosion_cooldown = 500  
                        
                    elif GLoss < bestGloss:
                        if explosion_cooldown > 0:
                            explosion_cooldown -= 1
                        else:
                        
                        
                            bestGloss = GLoss

                            save_layers('Wk_layers.npy', Wk)
                            np.save('Wo_final.npy', n.asnumpy(Wo_final))

                            np.save('trust.npy', n.array([trust], dtype=np.float32))
                            
                            save_layers('Wq_layers.npy', Wq)
                            save_layers('Wv_layers.npy', Wv)
                            save_layers('Wh1_layers.npy', Wh1)
                            save_layers('Wh2_layers.npy', Wh2)

                            save_layers('mWq_layers.npy', mWq)
                            save_layers('mWk_layers.npy', mWk)
                            save_layers('mWv_layers.npy', mWv)
                            save_layers('mWh1_layers.npy', mWh1)
                            save_layers('mWh2_layers.npy', mWh2)
                            np.save('mWo.npy', n.asnumpy(mWo))

                            save_layers('vWq_layers.npy', vWq)
                            save_layers('vWk_layers.npy', vWk)
                            save_layers('vWv_layers.npy', vWv)
                            save_layers('vWh1_layers.npy', vWh1)
                            save_layers('vWh2_layers.npy', vWh2)
                            np.save('vWo.npy', n.asnumpy(vWo))

                            np.save('adm_t.npy', np.array([t]))

                            # cpu_dict = {word: vectors[dictionaryLookup[word]].get() for word in words}
                            # np.save('vocab.npy', cpu_dict)
                            cpu_all = vectors.get()
                            cpu_dict = {word: cpu_all[dictionaryLookup[word]] for word in words}
                            np.save('vocab.npy', cpu_dict)

                            print(f"\n[Checkpoint] Step {i} | GLoss: {GLoss:.4f}")
                

                

            # learning_rate = max(0.00001, base_lr * (0.95 ** (i / (steps*0.01))))

            # warmup_steps = 1000'
            warmup_steps = 1000
            min_lr = 0.00001
            if(trustEngaged):
                trust = float(n.clip(n.array([trust]), 0.001, 1.5)[0])
            if i < warmup_steps:
                learning_rate = min_lr + (base_lr - min_lr) * (i / warmup_steps)
            else:
                progress = (i - warmup_steps) / (steps - warmup_steps)
                learning_rate = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

            curve = min(max(1, ((((i-5000)/10000) * math.sqrt(num_layers)))), math.sqrt(num_layers))

            

            
        
        print("\n" + str(GLoss/steps))
        


        

        


        
    def write(segment_str, leng):
        output_str = segment_str
        segment = segment_str.replace('-', " - ").replace('.', " . ").replace(',', " , ").replace('?', " ? ").replace('!', " ! ").replace(':', " : ").replace(';', " ; ").replace('--', " -- ").replace("'", " ' ").replace('"', ' " ').replace('(', " ( ").replace(')', " ) ").replace('[', " [ ").replace(']', " ] ").replace('—'," — ").replace('”', " ” ").replace('–', ' – ').replace(' s ', ' s ').replace('“', ' “ ').lower().split()
        
        tkns = []
        for word in segment:
            tkn = translate(word, merges)
            for l in tkn:
                if l in dictionaryLookup:
                    tkns.append(l)

        pad = dictionaryLookup.get('<PAD>', 0)

        # tkns_idx = [dictionaryLookup[t] for t in tkns]

        # if len(tkns_idx) < dim:
        #     pad_idx = dictionaryLookup.get('<PAD>', 0)
        #     tkns_idx = [pad_idx] * (dim - len(tkns_idx)) + tkns_idx

        segmentIdxs = [dictionaryLookup.get(t, pad) for t in tkns]

        if len(segmentIdxs) < dim:
            segmentIdxs = ([pad]) * (dim - len(segmentIdxs)) + segmentIdxs


        
        for w in range(leng):
            context = segmentIdxs[-dim:]
            seqlen = len(context)
            bSize = 1

            # if os.path.exists('Wo_' + str(id) + '.npy') and os.path.getsize('Wo_' + str(id) + '.npy') > 0:
            #     Wq = np.load(f'Wq_{id}.npy')
            #     Wk = np.load(f'Wk_{id}.npy')
            #     Wv = np.load(f'Wv_{id}.npy')
            #     Wo = np.load(f'Wo_{id}.npy')
            # else:
            #     Wq = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(dim, int(dim / num_heads)))
            #     Wk = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(dim, int(dim / num_heads)))
            #     Wv = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(dim, int(dim / num_heads)))
            #     Wo = np.random.uniform(-1/math.sqrt(dim), 1/math.sqrt(dim), size=(int(dim / num_heads), len(dict.keys())))

            
            # X = n.array([dict[y] for y in segment], copy=True) Apparently this is slow for cupy: bc it goes one by one, now i have to fix...
            # indexes = n.array([dictionaryLookup[word] for word in segment])

            # X = vectors[indexes].copy() + PE #when i did not copy it, it would change the actual vectors when i used the actual dict but now im still doing it even tho it may not be necessary
            

            
            # for i in range(len(X)):

            #     for j in range(dim):

            #         if(j % 2 == 0):

            #             X[i][j] += math.sin(i / (10000**((2*j) / dim)))

            #         else:

            #             X[i][j] += math.cos(i / (10000**((2*j) / dim)))


        


            #Previous Single Head Attention Trial
            # Q = X @ Wq
            # K = X @ Wk
            # V = X @ Wv

            # length = X.shape[0] #size of dim1
            # mask = n.tril(n.ones((length, length))) #take the lower triangle made up of ones of size base dim1 and height dim1

            # scores = (Q @ K.T) / math.sqrt(dim)

            # scores = n.where(mask == 0, -1e9, scores)

            # A =softmax(scores)
            
            # Z = A @ V



            # Z = normLayer(Z + X, 1e-5) #residual connection: dekut-dsail.github.io/tutorials/transformer-architecture/6.%20Layer%20Normalisation%20&%20Residual%20Connection.html

            # output = Z[-1] @ Wo
            #Previous Single Head Attention Trial ^

            #Attempting MultiHead



















            # lengthOfSegment = X.shape[0] #rows of X aka # of words
            # lengthOfSegment = dim
            
            dimPerHead = int(dim / num_heads)

            # Q = X @ Wq
            # K = X @ Wk
            # V = X @ Wv

            # Q = Q.reshape(lengthOfSegment, num_heads, dimPerHead) #basically we are splitting it into four arrays
            # V = V.reshape(lengthOfSegment, num_heads, dimPerHead)
            # K = K.reshape(lengthOfSegment, num_heads, dimPerHead)

            # Q = n.transpose(Q, (1,0,2)) #swap the first two dimensions Heads go first and then segment length, we do this bc after the Q = X @ Wq, our shape is (segment, dim) so even after our reshape, we wont have heads first which is y we must swap
            # K = n.transpose(K, (1,0,2))
            # V = n.transpose(V, (1,0,2))

            # scores = (Q @ n.transpose(K, (0,2,1))) / math.sqrt(dimPerHead) #swap the last two to do .T for the K. basically we are following the attenion mechanism formula


            # mask = n.tril(n.ones((lengthOfSegment, lengthOfSegment))) #same as last time: its to hide the future words in the weights
            # scores = n.where(mask == 0, -1e9, scores)

            # A = softmax(scores)

            # Z = A @ V #u get the shape (heds, length, dim) but we will now need to go back and undo this 

            # Z = n.transpose(Z, (1,0,2)) #swap the first two dims to get len first

            # Z = Z.reshape(lengthOfSegment, dim) #combine all of the heads to get one large vector


            
            # Z = Z + X

            # Zb = Z
            # Z = normLayer(Z, 1e-5)

            # hidden1 = Z @ Wh1 #got rid of Z[-1] bc this way it trains more than it used to by training off of all of the words



            # hidden1 = n.maximum(0, hidden1)
            # global Counterx
            # if(Counterx % 1000 == 0):
            #     active_neurons = n.sum(hidden1 > 0) / hidden1.size
            #     print(f" | ReLU Activity: {active_neurons:.2%}")
            # Counterx+=1

            # hidden2 = hidden1 @ Wh2

            # hidden2 += Z
            # h2b = hidden2
            # hidden2 = normLayer(hidden2, 1e-5)

            # output = hidden2 @ Wo

            input_arr = n.array([context])

            


            # tokens = []
            # for word in segment:
            #     tokens.extend(translate(word, merges))

            
            # pad = dictionaryLookup.get('<PAD>', 0)
            # token_ids = [dictionaryLookup.get(tok, pad) for tok in tokens]

            # seqlen = len(token_ids)

            embds = vectors[input_arr].copy()

            pos = PE[:seqlen, :]

            currentX = embds + pos
            
            mask = n.tril(n.ones((seqlen, seqlen))) 


            for i in range(num_layers):
                X = normLayer(currentX, 1e-5)

                Q = X @ Wq[i]
                K = X @ Wk[i]
                V = X @ Wv[i]

                Q = Q.reshape(bSize, seqlen, num_heads, dimPerHead) 
                V = V.reshape(bSize, seqlen, num_heads, dimPerHead)
            
                K = K.reshape(bSize, seqlen, num_heads, dimPerHead)

                # Q = n.transpose(Q, (1,0,2)) 
                # K = n.transpose(K, (1,0,2))
                # V = n.transpose(V, (1,0,2))

                Q = n.transpose(Q, (0,2,1,3)) 
                V = n.transpose(V, (0,2,1,3))
                K = n.transpose(K, (0,2,1,3))

                scores = (Q @ n.transpose(K, (0,1,3,2))) / math.sqrt(dimPerHead) 


                scores = n.where(mask == 0, -1e9, scores)

                A = softmax(scores)


            

                Z = A @ V 

                # plt.figure(figsize=(8, 6))
                # sns.heatmap(A, cmap='viridis', annot=False)
                # plt.title("Attention Heatmap (Matrix A)")
                # plt.xlabel("Key Tokens")
                # plt.ylabel("Query Tokens")
                # plt.show()

                Z = n.transpose(Z, (0,2,1,3)) 

                Z = Z.reshape(bSize, seqlen, dim)

                currentX = currentX + (Z / math.sqrt(num_layers))
                # global curve
                # Z = (Z / math.sqrt(num_layers)) * curve + currentX

                # Zn = normLayer(Z, 1e-5)





                X = normLayer(currentX, 1e-5)

                hidden1 = X @ Wh1[i] 

                # hidden1 = n.maximum(hidden1 * 0.01, hidden1) #LEAKY RELU: THE NEW AMAZING FIX to my "ubiquitous" dying nueron problem: sry but i learned too many new words for the sat and i had to use it.

                hidden1 = gelu(hidden1)

                

                

                hidden2 = hidden1 @ Wh2[i]


            
                currentX = currentX + hidden2


                


                

            out = currentX @ Wo_final

            logits = out[0,-1,:] / 0.3
            
            # for word in ['and</w>', 'the</w>']:
            #     logits[dictionaryLookup[word]] -= 0.3
            
            Probabilities = softmax(logits)
            
            sorted_indices = n.argsort(Probabilities)[::-1]
            sorted_probs = Probabilities[sorted_indices]
            cumulative_probs = n.cumsum(sorted_probs)
            
            cutoff = 0.9
            keep_idx = n.where(cumulative_probs <= cutoff)[0]
            keep = max(int(keep_idx[-1]) + 1 if len(keep_idx) > 0 else 1, 5)
            
            top_indices = sorted_indices[:keep]
            top_probs = sorted_probs[:keep]
            top_probs /= n.sum(top_probs)

            choice = n.random.choice(top_indices.flatten(), size=1,p=top_probs.flatten())
            next_idx = int(choice[0])
            next_word = words[next_idx]

            # k=10

            # remove = logits < n.partition(logits, -k)[-k]
            # logits[remove] = -n.inf

            # logits = logits / 0.7
            # penalty = 1.15
            # for tok_id in set(segmentIdxs):
            #     if logits[tok_id] > 0:
            #         logits[tok_id] /= penalty
            #     else:
            #         logits[tok_id] *= penalty


            # Probabilities = n.exp(logits) / n.sum(n.exp(logits))
            
            # next_idx = int(n.random.choice(a=len(Probabilities),size=1,p=Probabilities)[0])
            # next_word = words[next_idx]

            # segmentIdxs.append(next_idx)

            

            if next_word.endswith('</w>'):
                output_str += next_word[:-4] + " "
            else:
                output_str += next_word
            
            sys.stdout.write('\r' + output_str)
            sys.stdout.flush()

            
    x = "I Like Building"
    while x != "RETURN":
        x = input()
        write(x, 100)
        print("\n")


                

    #training STAGE 1: CONTEXTUALIZE WORD MATRIX^


'''
Log: 
  - 12/21/25
  - I am starting to make this model: embedding init is done.
  - The project might fail but this has been the dream for over a year: hopefully i learn

  - 12/22/25
  - I am starting to make my transformer: learning how attention mechanism works

  -12/24/25 
  - I have done it: made the base architecture: its not perfect but woohoo

  - 12/26/25
  - Yesterday and today i have been experimenting and been adding new guardrails since I hit new errors almost every training run, one less error means one step closer to success

  - 12/27/25
  - I am planning on doing my first large scale training run, it will last 17 hours: i want to max out my single head attention: even tho it may seem like a waste of time, I would like to see y ppl do not use this architecture before switching. This run will be from 5 pm today til 11 am tmr: hopefully my gpu does not melt

  - 12/28/25
  - Training run completed and I stopped it at 28% done: this left me with 6.55 GLoss which is horrible since I started the run at 6.66, so I have decided to switch architectures
  - I implemented the multi head and have left it at 8 heads bc my original of four was cool but i thought and realized that i should probably use more if my pc can handle it
  - I prefer to train using the massive 20 mil step loop 
     + This is bc I found that even if I stop early, it still trains a good amount and will save weights
     + Furthermore I normally stop at around 2% and do some tweaks such as decreasing/increasing lr and so on
     + In addition, I believe u should lock ur dictionary at first and only unlock when u believe that it has hit a wall: this prevents the model from constantly having to guess:
         * Will automate in coming days...
     + Also sometimes I feel that learning rate is not high enough so i just restart my training so it starts against at 0.001 or such bc the decay may be too aggressive -> this also allows for it to jump out of plateaus
  - I got tired of seeing it plateau at 6.49 even with multi head so i added ReLU -> its actually rly simple all u do is make all of the negative numbers zero which is important since it makes it nonlinear 

  - 12/30/25
  - Ive been working on the program for a few days...notable changes include switching from top k to top p bc I found that top k would include nonsense words when we should only be dealing with good words
     + Also bc I just found out abt top p...but anyways I also switched the input file from the old capitalism critiques to children's stories in hopes of achieving a more trainable ai since I realized that I cant even guess the next word in that text so that is on me.
     + Furthermore, I realized that I did + PE after norm in the attentionhead func and since i had already trained so much like that I just changed the generator func to be like that (yes i know its bad practice but I did not want to lose my progress)
     + Also I hit 4.29 GLoss YAYYYY!!!!! After like hours of training but still: i had low hopes for this model (especially this version since we do the loss on every word in the 512 training segment) but woohoo.
        * Also I would like to report that it generates somewhat coherent results "Once upon a time,  there was a big girl named there was a time there was a little girl named lucy she was very excited to explore one day to try it on an old she decided to see her teddy she was a beautiful and played for her room and tried to help" with the "once upon a time," being the only seed text
  - Also it is slowing down at this point and ReLU usage is up from the 49% it used to be at to a high 69.5ish% where it oscillates until rising steadily so I am thinking about implementing decay tommorow. 
  - Furthermore, I demoed this for the first time and while it gave some output which was not too amazing but it works.

  - 1/1/26
  - New year means new architecture: im gonna start and hopefully finish a method to make it multilayer: essentially im just gonna make functions for forward and backward and that should do the trick. Edit: Nevermind, although I can make it multilayer today, it wont be great and id have to transition my stuff to a new system where each weight is stored in an array of length layer.

  - 1/4/26
  - I actually went back on my word and made it multi-layer and now it has 6 layers. ReLU is being kept in a good range so I think we are good. 0.92...

  - 1/6/26
  - I know school has started again, I am focused on my schoolwork, but I left my training on until I got home today at 6 pm and I started the training yesterday at around 6:30ish pm. Longest training run of the whole model's life: i left dict unlocked however I have stopped that run now and locked the dict again to train on a more stable architecture.


  - 1/19/26
  - Its been a long time, it might seem like I gave up or something but I actually did the exact opposite, although school has been giving lot more homework since the AP tests are in a couple months, I spent the entirety of this long weekend building out the batching system. Yes, I know that it was made a long time ago its just that it had a ton of errors.
  - So changes:
        + Batch-based training and bug fixes: can u believe that it did not load any weights every time bc i had a file named incorrectly.... I spent like four hours trying to figure out why it did not "save"
        + Anyways, it works now and umm lets just say that it is soooooo slow when im using a batch greater than 1, so Ive just been running it with a batch size of one but Ill change it later and see what benefits there are :)
        + Furthermore, I would like to say that the old code from 1/6/26 was rly rly rly important for comparing my current to bc that one works like a charm while this one was kinda cooked for a long time.
            **** We HIT 5.79 GLoss ----> I know that is worse than before but this was under a couple hours so YAYYYY!!!!
        + Also .92 is holding up at 12 layers, 768 dimensions, and 8 heads. Can u believe it???!??!? Thats almost GPT2-small

  - 1/20/26
  - They should use training a transformer as some kind of anger control therapy: i left it to train for 24 full hours and when I chceked on it today after getting home after robotics, I saw that the GLoss which was 5.71 when I left had jumped all the way up to 7.47 bc apparently my word vectors had exploded. 
  - Bro there has to be some kind of software to just find the issues for me before I go all in and waste my time.... Hey thats a good idea.
  - Plus write is broken rn but we will fix that when we get to it. 
        + However, this is still so much more fun than making video games since Unity Errors are so vague you might as well give up. 
        + HEY!!! Ten minutes into training and we already hit 7.47

  - 1/25/26
  - Ok so big issue: I found that the larger model (ie 12 layer, 1024 dim, 16 head) is unable to break 5.9 in GLoss: I believe that this is likely caused by overfitting due to the large size so I hope to move to open web text 2: although I dont even know where to start with formatting that into a text file. 
  - Things to research: max size of a .txt file: i may have to switch to a different input data metric...
  - Worst case scenario is that I completely broke the entire system when I tried to build out the more complicated features like final output weighting instead of per layer so hopefully that is not the case...

  - 2/9/26
  - First log in a long time: basically while I was working on this project in the past two weeks, I did not have much time to really implement any new major changes. 
  - However, yesterday I added in punctuation to beta but no post processing. Also this has not been tested fully yet but it is at a GLoss of 5.01 as of 11:14 am today which means that after like one more day of training I think I will have good quality or at least slightly comprehensible output.

  - 2/28/26
  - Wow, a long time: well basically I have found some interesting news: apparently it works better without any decay... This is sad because apparently 0.92 was doing nothing but ruining the weight magnitude so it was the main issue: the relu being around 50% was just a weird side effect that I have yet to research fully but I will research it completely in alpha as that was the version with the most "benefit"
        - I would have worked on it more but swim started.
  - Anyways in other news: I added in a brand new input.txt made up of scrappings of the web from the FineWeb dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/CC-MAIN-2013-20/train?p=19
        - Just found out that "FineWeb" is anything but fine. Literally 25% of it is like website urls and copyrights which makes sense but they hinder convergence.
  - In other news, I added in a Graph to visualize the loss curves which can be seen in the loss_curve.png that will be created.
  - Also i found that its really difficult to break into 5 GLoss: especially on this dataset but that just means that we have found a new challenge to overcome

  
  - 3/4/26
  - Longest training run I have ever done just ended at 6.55 GLoss... Sad but thats how it is ig
  - Here are its final outputs: 
       - Output "I like building  , michael with the years below . a budget and dawn writer : real repair area . " " " " founder of time he is that always 20 a one as zero ( language are area , all their two nearly exist this for these that or those about a lot still on your and he said . ok . when they with rent this training when may record . there are 1900 that climate german on quantity firm . this sugar night as decent and human little make basic having focused as important 10 : 0 . close"
       - "I like building  , up involved . " " " " " " " " , hence give it , this … . 40 new numbering ( also is still number ) , these sale that the u . all . we have an information in any + . these horse ) 25 for all wednesday , in across one craft are necessary 2007 more corrupted at a spat ( 35 . now my notes . retrieved thesis your issues " " december in danger least . by halloween nothing information everyone better reporting . , it all , we've high austen from"
       - "I like building  for the whole buns print for santa conflict . in washington does about community . 5 to instance also risks . generally . s a version being and an required in in fine . ] its aquatic upon as well as much fund . " the landing toward days . as allow this is nothing . in local schools / . danger . and unhelpful eu . wade . a very year , retailers . this of each through all law . with running of lansing hallows case ( funding administration member ( there with education of hdf life ,"

  - 3/8/26
  - Byte Pair Encoding IMPLEMENTED!!!!
  - Essentially now the number of tokens is capped at 50k meaning it starts out at a lower GLoss and learns faster. I just hit 6.14 in 600 steps... YAYYY!!!
  - Basically what it does is merge together characters into one token based on how frequently they appear together which means more useful tokens like running which used to be its own token is now like run and ing. 
  
  - 4/25/26
  - Adam implemented!!! 
  - Actually I am not really telling the truth with this log. I didn't implement Adam today. I actually built it back in March before my SAT but it didn't exactly work because it needed fine tuning so I gave up on it to focus on scaling up and finding better data so I have been training on WikiText-103 from Salesforce's huggingface.
  - Also I found this really cool version of my vocab.npy that basically converged like 5x as quickly on all of my models which is weird because I'm not exactly sure why. However, I feel that my next few logs will be a little later because I have AP exams coming up as well as finals.

  - 5/29/26
  - School is out so now I basically have all day to code. We are going to make some insane breakthroughs.
  - I also made a custom trust system which works a little bit. Also I am going to be leaving for like two weeks very soon but I will leave it training and see what I get. 

  - 6/19/26
  - I am back and disappointed. It plateaued at around 5.0 after 100k steps. I could have done so much better....

  - 6/24/26
  - I've done it. I have finally broke 4.0. Model settled at around 3.90 to 3.95 but generation is really bad. Like alarmingly bad meaning that it likely means the tokens in my training were poor. 
  - I also found out what made my vocab.npy work well. It only had 18k tokens meaning that it was no where near the 50k I had set.

  DISCLAIMER: I attempted to do the math and I have failed a bunch of times so it may not be perfect. Also, I did need help from articles to get some of the math for the backpropogation. The math for PE and Attention came directly from Attention is All You Need. 

'''
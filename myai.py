import cupy as n
import numpy as np
import math
import os
import sys

#globals

new = False
dictLock = True
train = True
dim = 512
vocabSize = 0
num_heads = 8
learning_rate = 0.01 #the lr is rly high but I did this so it will cause the model to jump out of plateaus.
base_lr = learning_rate

num_layers = 6



GLoss = 0 #to be deleted later

Counterx = 0 #to be deleted later

used = []
for i in range(num_heads):
    used.append(0)

#globals ^




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#utils

def normLayer(x, smallConst): #https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    mean = n.mean(x, keepdims=True, axis=-1)
    variance = n.var(x, keepdims=True, axis=-1)
    variance = n.maximum(variance, 1e-5)

    return ((x-mean)/n.sqrt(variance + smallConst)) #math.sqrt did not work

def backNormLayer(out, x, smallConst): # this function came from a patchwork of articles https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/, https://veitner.bearblog.dev/backprob-through-layernorm/,
    m = n.mean(out, keepdims=True, axis=-1)

    v = n.var(x, keepdims=True, axis=-1)

    v = n.maximum(v, 1e-5)
    
    x_norm = normLayer(x, smallConst)

    dx = 1/(n.sqrt(v+smallConst)) * (out - m - x_norm * n.mean(x_norm * out, keepdims=True, axis=-1))

    return dx

def xinit(rows, cols): #https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
    limit = math.sqrt(6/(rows+cols)) * (1/math.sqrt(2*num_layers))
    return n.random.uniform(-limit, limit, size=(rows, cols))


def softmax(v): 
    exp_vector = n.exp(v - n.max(v, axis=-1, keepdims=True))
    probabilities = exp_vector / n.sum(exp_vector, axis=-1, keepdims=True)
    return probabilities
    

with open('input.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()



cleaned = raw_text.replace('-', " ").replace('.', "").replace(',', "").replace('?', "").replace('!', "").replace(':', "").replace(';', "").replace('--', " ").replace("'", " ").replace('"', " ").replace('(', " ").replace(')', " ").replace('[', " ").replace(']', " ").replace('—'," ").replace('”', " ").replace('–', ' ').replace(' s ', ' ').replace('“', ' ').lower().split()

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
for word, val in word_weights.items():
    ww[dictionaryLookup[word]] = val
#utils ^


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#training STAGE 1: CONTEXTUALIZE WORD MATRIX

def AttentionHead(segment, next_word, Wk, Wo, Wq, Wv, Wh1, Wh2): #segment length = dim
    
    cont = True
    output = ""
    if(len(segment) != int(dim)):
        cont = False

    if(cont):
        for i in range(len(used)):
            if(used[i] == 0):
                id = i
                break

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
        indexes = n.array([dictionaryLookup[word] for word in segment])

        X = vectors[indexes].copy() + PE #when i did not copy it, it would change the actual vectors when i used the actual dict but now im still doing it even tho it may not be necessary
        

        
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



















        lengthOfSegment = X.shape[0] #rows of X aka # of words
        
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
        all_A = []
        all_V = []
        all_Q = []
        all_K = []
        all_h1 = []
        all_Zb = []
        all_h2b = []

        currentX = X

        for i in range(num_layers):
            X = normLayer(currentX, 1e-5)
            all_X.append(X.copy())

            Q = X @ Wq[i]
            K = X @ Wk[i]
            V = X @ Wv[i]

            Q = Q.reshape(lengthOfSegment, num_heads, dimPerHead) 
            V = V.reshape(lengthOfSegment, num_heads, dimPerHead)
            K = K.reshape(lengthOfSegment, num_heads, dimPerHead)

            Q = n.transpose(Q, (1,0,2)) 
            K = n.transpose(K, (1,0,2))
            V = n.transpose(V, (1,0,2))

            scores = (Q @ n.transpose(K, (0,2,1))) / math.sqrt(dimPerHead) 


            mask = n.tril(n.ones((lengthOfSegment, lengthOfSegment))) 
            scores = n.where(mask == 0, -1e9, scores)

            A = softmax(scores)


            all_A.append(A.copy())
            all_V.append(V.copy())
            all_Q.append(Q.copy())
            all_K.append(K.copy())

            Z = A @ V 

            Z = n.transpose(Z, (1,0,2)) 

            Z = Z.reshape(lengthOfSegment, dim)

            Z = Z + currentX

            all_Zb.append(Z.copy())

            Zn = normLayer(Z, 1e-5)



            hidden1 = Zn @ Wh1[i] 

            hidden1 = n.maximum(0, hidden1)

            global Counterx
            if Counterx % 10001 == 0:
                
                active_ratio = n.mean(hidden1 > 0) 
                print(f" | Layer {i} ReLU Activity: {active_ratio:.2%}")
            Counterx+=1

            all_h1.append(hidden1)
            

            hidden2 = hidden1 @ Wh2[i]

            hidden2 += 0.5 *  Z
            all_h2b.append(hidden2.copy())

            h2b = hidden2
          
            if i == num_layers - 1:
                output = hidden2 @ Wo[i]
            else:
                currentX = hidden2









        
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------Forward Pass Above------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #Attempting MultiHead ^


        Probabilities = softmax(output)


        # i hit a probabilities are not non-negative error: so fix is below: DO NOT MAKE THIS MISTAKE AGAIN-> I had to restart which ruined an hour's worth of training
        Probabilities = n.clip(Probabilities, 1e-9, 1.0)
        Probabilities /= n.sum(Probabilities, axis=-1, keepdims=True) 



        target = n.array([dictionaryLookup[word] for word in segment[1:]] + [dictionaryLookup[next_word]]) #target is now every word except the first one

        weights = ww[target][:, n.newaxis] #finding weights for each word to encourage learning rarer words

        correct = Probabilities[n.arange(len(segment)), target] #basically this is the loss stuff its cool but only for visuals and this is the loss for the entire segment
 
        Loss = -n.mean(n.log(correct + 1e-10)) #loss for entire segment
        Error = Probabilities.copy()    

        Error[n.arange(len(segment)), target] -= 1.0

        # Error /= lengthOfSegment
        # Error *= weights

        rError = Error

        grad_Wq, grad_Wv, grad_Wk, grad_Wh1, grad_Wh2, grad_Wo = [None] * num_layers, [None] * num_layers, [None] * num_layers, [None] * num_layers, [None] * num_layers, [None] * num_layers
        for i in reversed(range(num_layers)):
            if i == num_layers - 1:
                grad_Wo[i] = (all_h2b[i].T @ Error) / lengthOfSegment
                dhidden2 = rError @ Wo[i].T
            else:
                dhidden2 = rError
            
            grad_Wh2[i] = (all_h1[i].T @ dhidden2) / lengthOfSegment
            dhidden1 = dhidden2 @ Wh2[i].T

            dhidden1[all_h1[i] <= 0] = 0

            grad_Wh1[i] = (normLayer(all_Zb[i], 1e-5).T @ dhidden1) / lengthOfSegment #New, u need this bc now the input is different from the normalized input while before it was not
            dz = backNormLayer(dhidden1 @ Wh1[i].T, all_Zb[i], 1e-5)
            dz += dhidden2
            dz = backNormLayer(dz, all_Zb[i], 1e-5)

            dz_heads = dz.reshape(lengthOfSegment, num_heads, dimPerHead)
            dz_heads = n.transpose(dz_heads, (1,0,2)) 


            dA = dz_heads @ n.transpose(all_V[i], (0,2,1)) 
            dV = n.transpose(all_A[i], (0,2,1)) @dz_heads 


            dSoftmax = all_A[i] * (dA - n.sum(dA * all_A[i], axis=-1, keepdims=True))
            dSoftmax /= math.sqrt(dimPerHead)
            dSoftmax = n.where(mask == 0, 0, dSoftmax)

            dQ = dSoftmax @ all_K[i]
            dK = n.transpose(dSoftmax, (0,2,1)) @ all_Q[i]

            dQ = n.transpose(dQ, (1,0,2)) 
            dK = n.transpose(dK, (1,0,2)) 
            dV = n.transpose(dV, (1,0,2)) 

            dQ = dQ.reshape(lengthOfSegment, dim) 
            dK = dK.reshape(lengthOfSegment, dim)
            dV = dV.reshape(lengthOfSegment, dim)


            
            grad_Wv[i] = all_X[i].T @ dV 
            grad_Wk[i] = all_X[i].T @ dK
            grad_Wq[i] = all_X[i].T @ dQ 

            
            Sx = dQ @ Wq[i].T + dK @ Wk[i].T + dV @ Wv[i].T

            rError = Sx + dhidden2
      

        

            # for i, word in enumerate(segment):
            #     vectors[word] -= (learning_rate * 0.01) * Sx[i]
            #     normalize = n.linalg.norm(dict[word])
            #     if normalize > 0:
            #         vectors[word] /= normalize
          


        for i in range(num_layers):
            for g in [grad_Wo[i], grad_Wv[i], grad_Wk[i], grad_Wq[i], grad_Wh2[i], grad_Wh1[i]]:

                if g is None:
                    continue
                norm = n.linalg.norm(g)
                if norm > 0.5:
                    g *= (0.5 / norm)
                g+= n.random.normal(0,0.001*base_lr, g.shape) 
                g -= n.mean(g, axis=0, keepdims=True)

            f = [grad_Wo[i], grad_Wv[i], grad_Wk[i], grad_Wq[i], grad_Wh2[i], grad_Wh1[i]]
            j = [Wo[i], Wv[i], Wk[i], Wq[i], Wh2[i], Wh1[i]]
            for w in range(len(f)):
                if f[w] is not None:
                    j[w] -= learning_rate * f[w]
            if(i%2 == 1):
                Wh1[i] *= 0.92
                Wh2[i] *= 0.92

        
        if(not dictLock):
            v_grad = n.clip(rError, -1.00, 1.0)
            vectors[indexes] -= (learning_rate * v_grad)

        used[id] = 0

        global GLoss
        GLoss += Loss
        return Wk, Wo, Wq, Wv, Wh1, Wh2


                


                

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
        # dSoftmax = A * (dA - n.sum(dA * A, axis=-1, keepdims=True))
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
               'Wo_layers.npy', 'Wh1_layers.npy', 'Wh2_layers.npy']


if all(os.path.exists(f) for f in layer_files):
    Wq_raw = np.load('Wq_layers.npy', allow_pickle=True)
    Wk_raw = np.load('Wk_layers.npy', allow_pickle=True)
    Wv_raw = np.load('Wv_layers.npy', allow_pickle=True)
    Wo_raw = np.load('Wo_layers.npy', allow_pickle=True)
    Wh1_raw = np.load('Wh1_layers.npy', allow_pickle=True)
    Wh2_raw = np.load('Wh2_layers.npy', allow_pickle=True)

    Wq = [n.asarray(layer) for layer in Wq_raw]
    Wk = [n.asarray(layer) for layer in Wk_raw]
    Wv = [n.asarray(layer) for layer in Wv_raw]
    Wo = [n.asarray(layer) for layer in Wo_raw]
    Wh1 = [n.asarray(layer) for layer in Wh1_raw]
    Wh2 = [n.asarray(layer) for layer in Wh2_raw]
else:
    Wq, Wk, Wv, Wo, Wh1, Wh2 = [], [], [], [], [], []
    for i in range(num_layers):
        Wq.append(xinit(dim, dim))
        Wk.append(xinit(dim, dim))
        Wv.append(xinit(dim, dim))
        Wh1.append(xinit(dim, 4 * dim))
        Wh2.append(xinit(4 * dim, dim))

        if i == num_layers - 1:
            Wo.append(xinit(dim, len(words))) 
        else:
            Wo.append(xinit(dim, dim)) 

    


def save_layers(filename, layer_list):
    obj_arr = np.empty(len(layer_list), dtype=object)
    for idx, layer in enumerate(layer_list):
        obj_arr[idx] = layer.get() 
    np.save(filename, obj_arr)
steps = 20000000
if(train):
    for i in range(steps):
        x = cleaned
        start = n.random.randint(int(dim + 1), len(x) - int(dim + 1) - 1)
        start = int(start)
        
        Wk, Wo, Wq, Wv, Wh1, Wh2 = AttentionHead(x[start:start+int(dim)], x[start+int(dim)], Wk, Wo, Wq, Wv, Wh1, Wh2)
        
        if i % 100 == 0:
            percent = (i + 1) / steps * 100
            bar = '█' * int(i / steps * 20)
            if(i == 0):
                lock = GLoss
            else:
                lock = GLoss/(i+1)
            sys.stdout.write(f'\rProgress: |{bar:<20}| {percent:.1f}% | {lock} | {dictLock}')
            sys.stdout.flush()

        if i % (steps * 0.0005) == 0:
            

            save_layers('Wk_layers.npy', Wk)
            save_layers('Wo_layers.npy', Wo)
            save_layers('Wq_layers.npy', Wq)
            save_layers('Wv_layers.npy', Wv)
            save_layers('Wh1_layers.npy', Wh1)
            save_layers('Wh2_layers.npy', Wh2)

            cpu_dict = {word: vectors[dictionaryLookup[word]].get() for word in words}
            np.save('vocab.npy', cpu_dict)
            print(f"\n[Checkpoint] Step {i} | GLoss: {GLoss/(i+1):.4f}")
   
    

        # learning_rate = max(0.00001, base_lr * (0.95 ** (i / (steps*0.01))))
        min_lr = 0.00001
        learning_rate = min_lr + (0.5) * (base_lr - min_lr) * (1 + math.cos(math.pi * i / steps))
        
    
    print("\n" + str(GLoss/steps))

    

    


    
def write(segment_str, leng):
    output_str = segment_str
    segment = segment_str.replace('-', " ").replace('.', "").replace(',', "").replace('?', "").replace('!', "").replace(':', "").replace(';', "").replace('--', " ").replace("'", " ").replace('"', " ").replace('(', " ").replace(')', " ").replace('[', " ").replace(']', " ").replace('—'," ").replace('”', " ").replace('–', ' ').replace(' s ', ' ').replace('“', ' ').lower().split()
    
    if len(segment) < dim:
        segment = ["<PAD>"] * (dim - len(segment)) + segment

    gen_words = []
    
    for w in range(leng):
        idx_array = n.array([dictionaryLookup.get(word, dictionaryLookup['<PAD>']) for word in segment])
        currentX = vectors[idx_array].copy()
        
        for i in range(num_layers):
            X_norm = normLayer(currentX, 1e-5)
            
            X_norm += PE
            
            lengthOfSegment = X_norm.shape[0]
            dimPerHead = int(dim / num_heads)

            Q = (X_norm @ Wq[i]).reshape(lengthOfSegment, num_heads, dimPerHead).transpose(1, 0, 2)
            K = (X_norm @ Wk[i]).reshape(lengthOfSegment, num_heads, dimPerHead).transpose(1, 0, 2)
            V = (X_norm @ Wv[i]).reshape(lengthOfSegment, num_heads, dimPerHead).transpose(1, 0, 2)

            scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(dimPerHead)
            mask = n.tril(n.ones((lengthOfSegment, lengthOfSegment)))
            scores = n.where(mask == 0, -1e9, scores)
            
            A = softmax(scores)
            Z = (A @ V).transpose(1, 0, 2).reshape(lengthOfSegment, dim)

            Z = Z + currentX
            Zn = normLayer(Z, 1e-5)

            h1 = n.maximum(0, Zn @ Wh1[i])
            h2 = h1 @ Wh2[i]

            h2 += Z * 0.5
            
            if i == num_layers - 1:
                out = h2 @ Wo[i]
            else:
                currentX = h2

        logits = out[-1] / 1.1
        
        for word in ['and', 'the']:
            logits[dictionaryLookup[word]] -= 0.5 
        
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

        next_idx = int(n.random.choice(top_indices.flatten(), size=1, p=top_probs.flatten()))
        next_word = words[next_idx]

        gen_words.append(next_word)
        segment.append(next_word)
        segment = segment[-dim:]
        output_str += " " + next_word
        
    print(output_str)

        
write("Lily found a ", 100)



            

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

  DISCLAIMER: I attempted to do the math and I have failed a bunch of times so it may not be perfect. Also, I did need help from articles to get some of the math for the backpropogation. The math for PE and Attention came directly from Attention is All You Need. 

'''

text = """
the cat sat on the mat
the dog sat on the rug
the cat ate fish
the dog ate meat
the bird sang loudly
the cat chased the bird
the dog chased the cat
"""

import numpy as np

words = text.lower().split()

vocab = sorted(set(words))

word_to_idx = {w:i for i,w in enumerate(vocab)}
idx_to_word = {i:w for w,i in word_to_idx.items()}

vocab_size = len(vocab)

tokens = [word_to_idx[w] for w in words]

seq_len = 3

X = []
Y = []

for i in range(len(tokens)-seq_len):
    X.append(tokens[i:i+seq_len])
    Y.append(tokens[i+seq_len])

X = np.array(X)
Y = np.array(Y)

def one_hot(i, size):
    x = np.zeros((size,1))
    x[i] = 1
    return x

class RNN:

    def __init__(self, input_size, hidden_size, output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = np.random.randn(hidden_size,input_size)*0.1
        self.Whh = np.random.randn(hidden_size,hidden_size)*0.1
        self.Why = np.random.randn(output_size,hidden_size)*0.1

        self.bh = np.zeros((hidden_size,1))
        self.by = np.zeros((output_size,1))

    def softmax(self,x):
        e = np.exp(x-np.max(x))
        return e/e.sum()

    def forward(self, sequence):

        h = np.zeros((self.hidden_size,1))

        self.cache=[]

        for token in sequence:

            x = one_hot(token,self.input_size)

            h = np.tanh(
                self.Wxh@x +
                self.Whh@h +
                self.bh
            )

            self.cache.append((x,h))

        y = self.Why@h+self.by

        p = self.softmax(y)

        return p,h

    def train_step(self, sequence, target, lr=0.01):

        p,h = self.forward(sequence)

        loss = -np.log(p[target]+1e-12)

        dy = p.copy()
        dy[target]-=1

        dWhy = dy@h.T
        dby = dy

        dWxh=np.zeros_like(self.Wxh)
        dWhh=np.zeros_like(self.Whh)
        dbh=np.zeros_like(self.bh)

        dh = self.Why.T@dy

        for x,h in reversed(self.cache):

            dtanh=(1-h*h)*dh

            dWxh += dtanh@x.T
            dbh += dtanh

            dWhh += dtanh@h.T

            dh=self.Whh.T@dtanh

        self.Wxh-=lr*dWxh
        self.Whh-=lr*dWhh
        self.Why-=lr*dWhy
        self.bh-=lr*dbh
        self.by-=lr*dby

        return loss.item()

class LSTM:

    def __init__(self,input_size,hidden_size,output_size):

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size

        H=hidden_size
        I=input_size

        self.Wf=np.random.randn(H,H+I)*0.1
        self.Wi=np.random.randn(H,H+I)*0.1
        self.Wc=np.random.randn(H,H+I)*0.1
        self.Wo=np.random.randn(H,H+I)*0.1

        self.bf=np.zeros((H,1))
        self.bi=np.zeros((H,1))
        self.bc=np.zeros((H,1))
        self.bo=np.zeros((H,1))

        self.Wy=np.random.randn(output_size,H)*0.1
        self.by=np.zeros((output_size,1))

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def softmax(self,x):
        e=np.exp(x-np.max(x))
        return e/e.sum()

    def forward(self,sequence):

        h=np.zeros((self.hidden_size,1))
        c=np.zeros((self.hidden_size,1))

        for token in sequence:

            x=one_hot(token,self.input_size)

            z=np.vstack((h,x))

            f=self.sigmoid(self.Wf@z+self.bf)
            i=self.sigmoid(self.Wi@z+self.bi)
            g=np.tanh(self.Wc@z+self.bc)
            o=self.sigmoid(self.Wo@z+self.bo)

            c=f*c+i*g
            h=o*np.tanh(c)

        y=self.Wy@h+self.by

        p=self.softmax(y)

        return p,h

rnn = RNN(vocab_size,64,vocab_size)

for epoch in range(1000):

    loss=0

    for seq,target in zip(X,Y):

        loss += rnn.train_step(seq,target,0.05)

    if epoch%100==0:
        print(epoch,loss)

def predict(model, sentence):

    seq = sentence.lower().split()

    ids = [word_to_idx[w] for w in seq]

    p,_ = model.forward(ids)

    idx = np.argmax(p)

    return idx_to_word[idx]

print(predict(rnn,"the cat sat"))
print(predict(rnn,"the dog ate"))
print(predict(rnn,"the cat ate"))
import numpy
import random

#dimensions, number of runs, threshold for dot product
n=30000
runs=1
thresh = 0.2

#HRR stuff
def cconv(a, b):
    '''
        Computes the circular convolution of the (real-valued) vectors a and b.
        '''
    return numpy.fft.ifft(numpy.fft.fft(a) * numpy.fft.fft(b)).real

def ccorr(a, b):
    '''
        Computes the circular correlation (inverse convolution) of the real-valued
        vector a with b.
        '''
    return cconv(numpy.roll(a[::-1], 1), b)

def cosine(a,b):
    '''
        Computes the cosine of the angle between the vectors a and b.
        '''
    sumSqA = numpy.sum(a**2.0)
    sumSqB = numpy.sum(b**2.0)
    if sumSqA == 0.0 or sumSqB == 0.0: return 0.0
    return numpy.dot(a,b) * (sumSqA * sumSqB)**-0.5

def normalise(a):
    '''
        Normalize a vector to length 1.
        '''
    return a / numpy.sum(a**2.0)**0.5

def generate_vectors(m,numb):
    for x in range(numb):
        symb = ''.join([random.choice(string.ascii_letters) for i in xrange(random.randint(1,12))])
        v = make_vector_symbol(symb)
        store(v,symb,m)

def generate_rand():
    symb = ''.join([random.choice(string.ascii_letters) for i in xrange(12)])
    return symb

def generate_symbol(dict):
    symb = ''.join([random.choice(string.ascii_letters) for i in xrange(12)])
    if (symb in dict.items()):
        while(symb in dict.items()):
            symb = ''.join([random.choice(string.ascii_letters) for i in xrange(12)])
    return symb

def make_vector_symbol(symbol):
    sigma = float(1.0/n)
    x = numpy.random.normal(0,sigma,n)
    x_norm = normalise(x)
    #cm[symbol] =  x_norm
    #s = cm[symbol]
#return s
    return x_norm

def generate_all(all,x):
    for i in range(0,x):
        gr = generate_rand()
        all.append(gr)
    return all

def add_pairs(cxn,x):
    for i in range(0,x):
        cxn.append(gr)

def rem_pairs(cxn,x):
    for j in range(0,x):
        cxn.pop()

def replace_pairs(cxn,x):
    cxn[i] = generate_rand()

def make_symbol_vector(vector,name,cm):
    x_norm = normalise(vector)
    cm[name] =  x_norm
    return cm[name]

def store(vec,name,mem):
    mem[name] = vec

def get_vector_symbol(symbol,cm):
    if symbol in cm:
        v = cm[symbol]
    else:
        make_vector_symbol(symbol)
        v = cm[symbol]
    return v

def add_to_structure(new_pair,str):
    return new_pair + str

def cleanup (x,mem):
    #set to a bit less than the poss min value
    highest=-2.0
    for key, val in mem.items():
    #cosine similarity (normalised dot-product)
        prod = cosine(x,val)
        if (prod>highest):
            highest = prod
            index = key
    return {'highest':highest, 'final':index}

def dereference(a,struct,mem):
    deref = ccorr(a,struct)
    final = cleanup(deref,mem)
    return final

def make_vocab(vocab):
    for i, el in enumerate(vocab):
        if el not in vocabulary:
            s = make_vector_symbol(el)
            vocabulary[el] = s
#print(vocabulary.keys())

def create_cxn(vocab,k):
    cxn = create_ini()
    mem = Dict()
    cleanups[k] = mem
    for z, el in enumerate(vocab):
        store(vocabulary[el],el,mem)
    for z in range(len(vocab)-1):
        if(z%2==0):
            pair = cconv(vocabulary[vocab[z]],vocabulary[vocab[z+1]])
            #print(vocab[z],vocab[z+1])
            store(pair,generate_symbol(mem),mem)
            cxn = pair+cxn
    return cxn

def create_cxn_set(cset):
    cxn_set = []
    #make cxns
    for i,cx in enumerate(cset):
        cxn = create_cxn(cx,i)
    #put them into a set
        cxn_set.append(cxn)
    return cxn_set

def questions(voc,ext,itn):
    av_dp=0.0
    mis = 0
    for i, it in enumerate(voc):
        if(i%2==0):
            final = dereference(vocabulary[it],ext,cleanups[itn])
            av_dp+=final['highest']
            if(final['final'] == voc[i+1]):
                mis+=1
    return {'dp':av_dp, 'mistakes':mis}

def write(data,writer):
    for el in data:
        writer.write(str(el))
    writer.write(" \n")

def generate_mistakes():
    for k in range(len(all)):
        for i in range(len(all[k])):
            if(i%2==0):
                mistakes[all[k][i]]=0.0

def get_length(voc):
    return len(voc)

def get_items_to_deref(items):
    deref_set=[]
    #print(items)
    for i in items:
        deref_set.append(i)
    return deref_set

prod_overall=0.0
mis_overall=0.0
#run the experiments
for run in range(runs):
    all=[[]]
    trans=[]
    generate_all(all[0],1000)
    for en,i in enumerate(all[0]):
        trans.append(all[0][en])

    #one vocabulary for all vectors
    vocabulary = Dict()
    #individual clean-up memory for each cxn/trans
    cleanups = Dict()

    make_vocab(trans)

    trans_set = create_cxn_set([trans])

    #dereference
    mis=0 #actually correct answer count

    for itn,item in enumerate([trans]):
        answer = questions(item,[trans_set],0)
            #dp+=answer['dp']
        mis+=answer['mistakes']
    #add up the mistakes for each run (mistakes by the number of pairs to deref)
    #prod_overall+=dp/(lng/2.0)
    mis_overall+=mis/(len(trans)/2.0)

print(mis_overall/float(runs))

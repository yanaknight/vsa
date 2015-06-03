import numpy
import random

#settings
n=3000 #vector dimensionality
runs=10 #runs per round
thresh = -1.0 #dot prod threshold
cl=2500 # number of vectors in cleanup
measure = 3.0 #are you using pairs or tripples?
num_um = 50 #set the number of units
num_pairs = 150 # e.g.15/3=5

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

#random vectors

def generate_vectors(m,numb):
    for x in range(numb):
        symb = ''.join([random.choice(string.ascii_letters) for i in xrange(random.randint(1,1120))])
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

#make a vector for a symbol
def make_vector_symbol(symbol):
    sigma = float(1.0/n)
    x = numpy.random.normal(0,sigma,n)
    x_norm = normalise(x)
    return x_norm

#symbol for vector
def make_symbol_vector(vector,name,cm):
    x_norm = normalise(vector)
    cm[name] =  x_norm
    return cm[name]

#store in a specified memory
def store(vec,name,mem):
    mem[name] = vec

#get vector for a symbol or make one in a memory
def get_vector_symbol(symbol,cm):
    if symbol in cm:
        v = cm[symbol]
    else:
        make_vector_symbol(symbol)
        v = cm[symbol]
    return v

def get_symb_vector(vector,cm):
    for k,v in cm.items():
        if vector.all() == v.all():
            return k

#generate random vectors
def generate_all(all,x):
    for i in range(0,x):
        gr = generate_rand()
        all.append(gr)
    return all

#replace random vectors with a unit name
def replace_unit(lst):
    gr = generate_rand()
    x=0
    for i in range(0,len(lst)):
        if(x==2):
            lst[i-2]= gr
            x=0
        else:
            x+=1
    return lst

#clean up after dereferencing
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

#create a construction by convolving and adding
def create_cxn(vocab,k):
    cxn = create_ini()
    mem = Dict()
    cleanups[k] = mem
    for z, el in enumerate(vocab):
        store(vocabulary[el],el,mem)
    x=0
    for z in range(len(vocab)):
        if (x==2):
            pair = cconv(vocabulary[vocab[z-2]],vocabulary[vocab[z-1]])
            pair1 = cconv(vocabulary[vocab[z]],pair)
            store(pair,generate_symbol(mem),mem)
            store(pair1,generate_symbol(mem),mem)
            cxn = pair1+cxn
            x=0
        else:
            x+=1
    return cxn

#create a set from a list of vectors
def create_cxn_set(cset):
    cxn_set = []
    #make cxns
    for i,cx in enumerate(cset):
        cxn = create_cxn(cx,i)
    #put them into a set
        cxn_set.append(cxn)
    return cxn_set

#dereferencing
def questions(voc,ext,itn):
    #print(item)
    av_dp=0.0
    cor = 0
    x=0
    for i, it in enumerate(voc):
        if (x==2):
            x=0
            conv = cconv(vocabulary[voc[i-2]],vocabulary[voc[i-1]])
            final = dereference(conv,ext,cleanups[itn])
            av_dp+=final['highest']
            if(final['final'] == voc[i]):
                cor+=1
        else:
            x+=1
    return {'dp':av_dp, 'mistakes':cor}

def write(data,writer):
    for el in data:
        writer.write(str(el))
    writer.write(" \n")

def get_length(voc):
    return len(voc)

#which items to dereference?
def get_items_to_deref(items):
    deref_set=[]
    #print(items)
    for i in items:
        deref_set.append(i)
    return deref_set

#run the experiments
prod_overall=0.0
prec_overall=0.0
for run in range(runs):
    all=[[]]
    trans=[]
    generate_all(all[0],num_pairs)
    #this replaces all first items with a name of the unit, can be done for several units
    i=0
    for j in range(0,num_um):
        all[0][i:(num_pairs/num_um+i)] = replace_unit(all[0][i:(num_pairs/num_um+i)])
        i+=num_pairs/num_um
        #print(num_pairs/num_um)
    
    for en,i in enumerate(all[0]):
        trans.append(all[0][en])

    #one vocabulary for all vetors
    vocabulary = Dict()
    #individual clean-up memory for each cxn/trans
    cleanups = Dict()
    lng=0
    
    make_vocab(trans) #make a vocab from the vectors
    trans_set = create_cxn_set([trans])
    generate_vectors(cleanups[0],cl-(len(cleanups[0])))
    #dereference
    corr=0 #correct answers
    dp=0.0  #dot prod
    #which items to dereference
    deref_set = get_items_to_deref(all)
    for itn,item in enumerate(all):
        if item in deref_set:
            lng += get_length(item)
            answer = questions(item,[trans_set],itn)
            dp+=answer['dp']
            corr+=answer['mistakes']
            #add up corr answers for each run
    prod_overall+=dp/(lng/measure) #overall dot prod
    prec_overall+=corr/(lng/measure) #precision score

print(prec_overall/float(runs))

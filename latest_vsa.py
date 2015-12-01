import numpy
import random
import string

#dimensions, number of runs, threshold for dot product
n=1000
runs=1
thresh = 0.2
cl=50 # number of vectors in cleanup

###small example grammar using the approach in units.py###############

###process:
###create a transient structure, split into units (because we don't know how to match parts of a str yet...)
###match each unit to all cxns, choose the one with the highest dot prod
###merge by adding the cxn to the trans str (unit), then add the units together
###deref values from new str (specify which elements you want to dereference)

def Dict(**args):
    """Return a dictionary with argument names as the keys,
        and argument values as the key values"""
    return args
#we begin with our construction set
all=[
      ['meaning','referent','string','the','sem-class','selector','lex-class','article'],
     ['meaning','mouse','string','mouse','sem-class','object','lex-class','noun','syn-function','nominal', 'number', 'sg','person','3'],
     ['meaning','green','string','green','sem-class','property','lex-class','adjective'],
     ]

#transient structure
trans = ['string','the','string','green','string','mouse']

def create_ini():
    structure = numpy.zeros(n)
    return structure

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
    return x_norm

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

#any construction with dot prod > thresh, should be applied and merged in (by addition)
def match_merge(trans_sets,cxn_set):
    trans = []
    max_products=[]
    added = 0
    for i in range(len(trans_set)):
        products = []
        for j in range(len(cxn_set)):
            prod=cosine(trans_set[i],cxn_set[j])
            added = added + cxn_set[j]
            products.append(prod)
        if(any(p >= thresh for p in products)):
            for pr in products:
                if (pr>thresh):
                    k = products.index(pr)
                    max_products.append(pr)
                    ext_trans = trans_set[i]+cxn_set[k]
                    trans.append(ext_trans)
    return {'struct':trans, 'prod':max_products}

def dereference(a,struct,mem):
    deref = ccorr(a,struct)
    final = cleanup(deref,mem)
    return final

def make_vocab(vocab):
    for i, el in enumerate(vocab):
        if el not in vocabulary:
            s = make_vector_symbol(el)
            vocabulary[el] = s

def create_cxn(vocab,k):
    cxn = create_ini()
    mem = Dict()
    cleanups[k] = mem
    for z, el in enumerate(vocab):
        store(vocabulary[el],el,mem)
    for z in range(len(vocab)-1):
        if(z%2==0):
            pair = cconv(vocabulary[vocab[z]],vocabulary[vocab[z+1]])
            store(pair,generate_symbol(mem),mem)
            cxn = pair+cxn
    return cxn

def create_cxn_set(cset):
    cxn_set = []
    #make cxns
    for i,cx in enumerate(cset):
        cxn = create_cxn(cx,i)
        cxn_set.append(cxn)
    return cxn_set

def questions(voc,ext,itn):
    av_dp=0.0
    cor = 0
    for i, it in enumerate(voc):
        if(i%2==0):
            final = dereference(vocabulary[it],ext,cleanups[itn])
            av_dp+=final['highest']
            #print(final['final'],voc[i+1])
            if(final['final'] == voc[i+1]):
                cor+=1
    return {'dp':av_dp, 'mistakes':cor}

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
prec_overall=0.0
#run the experiments
for run in range(runs):
    #make one vocabulary for all vetors
    vocabulary = Dict()
    #individual clean-up memory for each cxn/trans
    cleanups = Dict()
    lng=0
    for i in range(len(all)):
        make_vocab(all[i])
    
    make_vocab(trans)

    cxn_set=create_cxn_set(all)

    #make trans str split into units
    un1 = create_cxn(trans[:2],len(cleanups)+1)
    un2 = create_cxn(trans[2:],len(cleanups)+1)
    trans_set = []
    trans_set.append(un1)
    trans_set.append(un2)

    #match and merge!
    for f in range(1):
        ext = match_merge(trans_set,cxn_set)
        print([ext['prod']])
        trans_set=[]
        trans_set.append(ext['struct'][0])
        trans_set.append(ext['struct'][1])

#how to copy a new memory over? where do you deref from the second time?

    #dereference
    corr=0
    dp=0.0
    unit_cnt = 0
    #say which items you want to dereference
    deref_set = get_items_to_deref(all)
    for itn,item in enumerate(all):
        if item in deref_set:
            lng += get_length(item)
            #have to sum the units probably
            answer = questions(item,trans_set[0]+trans_set[1],itn)
            unit_cnt+=1
            dp+=answer['dp']
            corr+=answer['mistakes']

    #add up the mistakes for each run (mistakes by the number of pairs to deref)
    prod_overall+=dp/(lng/2.0)
    prec_overall+=corr/(lng/2.0)

print(prec_overall/float(runs),prod_overall/float(runs))

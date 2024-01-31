import io, math, re, sys

# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

# Find the most similar response in terms of token overlap.
def most_sim_overlap(query, responses):
    q_tokenized = tokenize(query)
    max_sim = 0
    max_resp = "Sorry, I don't understand"
    for r in responses:
        r_tokenized = tokenize(r)
        sim = len(set(r_tokenized).intersection(q_tokenized))
        if sim > max_sim:
            max_sim = sim
            max_resp = r
    return max_resp

# Code for loading the fasttext (word2vec) vectors from here (lightly
# modified): https://fasttext.cc/docs/en/crawl-vectors.html
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def sum_lists(*args):
    return list(map(sum, zip(*args)))

def response_vector_calc(word_vec,responses):
    t={}
    i=0
    result=[0]*300
    for res in responses:
        r_tokenize=tokenize(res)
##        n=len(r_tokenize)
        norm_vec={}
        n=0
        for word in r_tokenize:
            emb={}
            if word in word_vec.keys():
                emb[word]=word_vec[word]
                norm_den={}
                norm_term=0
                for v in emb[word]:
                    norm_term+=v**2
                norm_term=math.sqrt(norm_term)
                norm_den[word]=norm_term
                
                s=[]
                for v in emb[word]:
                    s.append(v/norm_den[word])

                norm_vec[word]=s
                n=n+1
##            else:
##                n=n-1

        if(n==0):
            n=len(r_tokenize)
        l=[]
        for i in norm_vec.values():
            l.append(i)

        for i in range(len(l)):
            a=result
            b=l[i]
            result=sum_lists(a,b)
        result[:] = [x / n for x in result]
        t[res]=result


    return t

        
    
def vec_calc(word_vec,query):
    t={}
    result=[0]*300
    q_tokenize=tokenize(query)
##    n=len(q_tokenize)
    norm_vec={}
    n=0
    for word in q_tokenize:
        emb={}

        if word in word_vec.keys():
            emb[word]=word_vec[word]
            norm_den={}
            norm_term=0
            for v in emb[word]:
                norm_term+=v**2
            norm_term=math.sqrt(norm_term)
            norm_den[word]=norm_term
            
            s=[]
            for v in emb[word]:
                s.append(v/norm_den[word])

            norm_vec[word]=s
            n=n+1
##        else:
##            n=n-1

    if(n==0):
        n=len(q_tokenize)
    l=[]
    for i in norm_vec.values():
        l.append(i)

    for i in range(len(l)):
        a=result
        b=l[i]
        result=sum_lists(a,b)
    result[:] = [x / n for x in result]
    t[query]=result
    return t





def most_sim(calc_query_vector, resp_vec,responses,query):
    max_sim = 0
    max_resp = "Sorry, I don't understand"
    norm_term=0

    for v in calc_query_vector[query]:
        norm_term+=v**2
    norm_term=math.sqrt(norm_term)
    query_mag=norm_term
    q_tokenize=tokenize(query)

    for res in responses:
        cosine_sim=0
        norm_term=0
        r_tokenize=tokenize(res)


        for v in resp_vec[res]:
            norm_term+=v**2
        norm_term=math.sqrt(norm_term)
        resp_mag=norm_term
        tot_mag= query_mag*resp_mag

        vec_mul=list(map(lambda x,y: x*y ,calc_query_vector[query],resp_vec[res]))
        summation=0
        for i in vec_mul:
            summation+=i
        if tot_mag==0:
            return 'Sorry, I dont understand'

        cosine_sim=summation/tot_mag
##                print(cosine_sim)
        if cosine_sim > max_sim:
            max_sim = cosine_sim
            max_resp = res
##    print(max_sim)
    return max_resp
        
        
        
        
 

if __name__ == '__main__':
    # Method will be one of 'overlap' or 'w2v'
    method = sys.argv[1]

    responses_fname = 'gutenberg.txt'
    vectors_fname = 'cc.en.300.vec.10k'

    responses = [x.strip() for x in open(responses_fname)]

    # Only do the initialization for the w2v method if we're using it
    if method == 'w2v':
        print("Loading vectors...")
        word_vectors = load_vectors(vectors_fname)
        # Hint: Build your response vectors here
        resp_vec=response_vector_calc(word_vectors,responses)
        
##        print(response_vec)
        

    print("Hi! Let's chat")
    while True:
        query = input()
        if method == 'overlap':
            response = most_sim_overlap(query, responses)
        elif method == 'w2v':
            calc_query_vector = vec_calc(word_vectors,query)
            response=most_sim(calc_query_vector, resp_vec,responses,query)

        print(response)

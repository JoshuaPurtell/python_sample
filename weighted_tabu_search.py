## Has Python Code for Connectivity Algos
# Eventually will run this in C++

## Weighted Tabu Search
#Given Adjacency Matrix A, Additional Connection Matrix P, and
#limit k, and val limit W, find approx for optimal L2

#Kincaid 2008, Vargo 2010 for unweighted

import networkx as nx

import numpy as np

import random

import copy


##Graph Utilities

#follows,DAO_memberships are dictionaries with addresses as keys
#lmbda gives weighting ration btwn follows and DAO memberships, eventually
#will be more complex

def add_weighted_edges(G,user_hash,follows,DAO_memberships,lmbda):
    for user in follows.keys():
        if user == user_hash:
            continue
        #check if edge exists 
        if [user_hash,user] not in G.edges and [user,user_hash] not in G.edges:
            #initialize edge
            G.add_edge(user_hash, user, weight=0)

            #now, if follows, add lmbda weight to edge
            if user in follows[user_hash]:
                G.edges[user_hash,user]['weight'] += lmbda
            if user_hash in follows[user]:
                G.edges[user_hash,user]['weight'] += lmbda
            
            #now, if share DAO, add 1 weight to edge
            intersection = [i for i in set(DAO_memberships[user_hash]) if i in set(DAO_memberships[user])]
            if len(intersection) > 0:
                G.edges[user_hash,user]['weight'] += 1.0
        else:
            continue
        
    return G


def create_G(follows,DAO_memberships,wf,lmbda):
    #initialize graph
    G = nx.Graph()
    #add a node for each address
    users = follows.keys()
    G.add_nodes_from(users)

    #now, add edges and assign weights according to function that's passed
    for user in users:
        G = wf(G,user,follows,DAO_memberships,lmbda)
    return G


def create_A(G,follows):
    A = np.zeros((len(G.nodes),len(G.nodes)))

    users = follows.keys()
    for n_index in range(len(G.nodes)):
        if n_index < len(G.nodes)-1:
            for m_index in range(n_index+1,len(G.nodes)):
                #check if there's an edge
                #print(G.nodes)
                #print("mn",n_index," ",m_index)
                if [users[n_index],users[m_index]] in G.edges:
                    A[n_index,m_index] = G.edges[users[n_index],users[m_index]]["weight"]
                    A[m_index,n_index] = G.edges[users[n_index],users[m_index]]["weight"]
    return A

def create_L(G,A):
    L = np.zeros((len(G.nodes),len(G.nodes)))
    for i in range(len(G.nodes)):
        for j in range(len(G.nodes)):
            if i==j:
                L[i,j] = sum(A[:,j])
            else:
                L[i,j] = - A[i,j]
    return L

#find how to get second eigenvalue and fiedler vector

def eig_fiedler(L):
    w,v = np.linalg.eig(L)
    wl = [w[e] for e in range(len(w))]
    wl.sort(reverse=False)
    alg_con = wl[1]
    #print("eigenvalues"," ",wl)
    fiedler_vector = v[:,wl.index(alg_con)]
    
    return alg_con,fiedler_vector

## Modified Greedy Perturbation Algorithm

def argmax(Z):
    maxi = -1
    maxz = -10e90
    for i in range(len(Z)):
        if Z[i] > maxz:
            maxi = i
    return maxi

#P = [[1,4],[3,4]], etc
def MGPA(G,A,L,P,follows,k):
    eig,Fiedler = eig_fiedler(L)
    
    for ik in range(k):
        if len(P)==0:
            continue
        #use eq 16
        DP = [0 for p in P]
        for ip in range(len(P)):
            i,j = P[ip][0],P[ip][1]
            DP[ip] = A[i,j]*(Fiedler[i]-Fiedler[j])**2
        #select possible edge with maximum value
        di = argmax(DP)
        Gn = [e for e in G.nodes]
        if [Gn[P[di][0]],Gn[P[di][1]]] in G.edges:
            #print("adding")
            ##TODO: accomodate both changes in follows and in DAO membership
            G.edges[Gn[P[di][0]],Gn[P[di][1]]]['weight'] += 1
        else:
            G.add_edge(Gn[P[di][0]],Gn[P[di][1]], weight=1.0)
        P.pop(di) #eliminate chosen edge from candidates
    
    #get updated lmbda
    A_star = create_A(G,follows)
    L_star = create_L(G,A_star)
    eig,Fiedler = eig_fiedler(L_star)
    return G,eig

def Ns_gen(s,p,P,G):
    Gn = [e for e in G.nodes]
    vi,vj = s[p][0],s[p][1]
    Ns = []
    for pi in range(len(P)):
        if vi in P[pi] or vj in P[pi]:
            Ns.append([Gn[P[pi][0]],Gn[P[pi][1]]])
    
    #now, for random element
    admissible = False
    proposal_i = 0
    idxs = [i for i in range(len(P))]
    random.shuffle(idxs)
    
    while admissible == False:
        #need clause to escape when there's no admissible sol'n
        if proposal_i == len(idxs):
            admissible = True
            rc = random.choice(P)
            Ns.append([Gn[rc[0]],Gn[rc[1]]])
        elif (vi in P[idxs[proposal_i]] or vj in P[idxs[proposal_i]]):
            proposal_i+=1
        else:
            Ns.append([Gn[P[idxs[proposal_i]][0]],Gn[P[idxs[proposal_i]][1]]])
            admissible = True
    return Ns

def lmbda2(G,s,follows):
    G = add_edges_from_s(G,s)
    A = create_A(G,follows)
    L = create_L(G,A)
    eig,_ = eig_fiedler(L)
    return eig

def restore_P(P,s_old):
    for e in s_old:
        P.append(e)
    return P


##TODO: fix
def new_Tabu(T,s,s_prime):
    if len(T)<len(s):
        Tprime = s[0:(len(T)-1)]
    else:
        Tprime = s.append(T[0:(len(T)-1-len(s))])
    return Tprime

def add_edges_from_s(G,s_star):
    Gn = [e for e in G.nodes]
    edges = [[Gn[s[0]],Gn[s[0]]] for s in s_star]
    G.add_edges_from(edges) #weighted?
    return G


def WTS(G,P,k,Tabu_length,Phi,follows):
    
    #randomly choose k edges from P to find s_0
    s_0 = []
    for ik in range(k):
        #print("P ",P)
        ix =random.choice([i for i in range(len(P))])
        s_0.append(P[ix])
        P.pop(ix) #randomly choose from P
    s = s_0
    s_star = s_0
    lmbda2_star = 0
    T = [[-1,-1] for i in range(Tabu_length)]


    for iphi in range(Phi):
        Ns = [[] for i in range(k)]
        for ip in range(k):
            Ns[ip] =  Ns_gen(s,ip,P,G)
        ixs = range(len(Ns))
        random.shuffle(ixs)
        s_prime = [P[ixs[ip]] for ip in range(k)]
        eig = lmbda2(G,s_prime,follows)
        if eig > lmbda2_star:
            P = restore_P(P,s)
            #add to Tabu
            T = new_Tabu(T,s,s_prime)
            s = s_prime
            s_star = s_prime
            lmbda2_star = eig
        else:
            T = new_Tabu(T,s,s_prime)
            s = s_prime
    #convert s_star
    G = add_edges_from_s(G,s_star)
    return lmbda2_star,G

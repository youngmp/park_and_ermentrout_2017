"""
library file
"""

import numpy as np

import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier


def collect_disjoint_branches(diagram,all_sv=True,return_eval=False,sv_tol=.1,remove_isolated=True,isolated_number=2,remove_redundant=True,redundant_threshold=.01,N=20,fix_reverse=True):
    """
    collect all disjoint branches into disjoint arrays in a dict.



    diagram.dat: all_info.dat from xppauto version 8. currently not compatible with info.dat. also need multiple branches. will not collect single branch with multiple types.
    recall org for xpp version 8:
    type, branch, 0, par1, par2, period, uhigh[1..n], ulow[1..n], evr[1] evm[1] ... evr[n] evm[n]
    yes there is a zero there as of xpp version 8. I don't know why.

    for more information on how diagram is organized, see tree.pdf in the xpp source home directory.

    all_sv: True or False. in each branch, return all state variables (to be implemented)
    return_eval: return eigenvalues (to be implemented)
    sv_tol: difference in consecutive state variables. If above this value, break branch.
    remove_isolated: True/False. If a branch has fewer than isolated_number of points, do not include.
    isolated_number: see previous line

    remove_redundant: if branches overlap, remove. we require the max diff to be above redundant_threshold
    by default, we keep branches with a longer arc length.
    N: number of points to check for redundancy 

    fix_reverse: True/False. some branches are computed backwards as a function of the parameter. If so, reverse.


    
    """


    

    # get number of state variables (both hi and lo values, hence the 2*)
    varnum = 2*len(diagram[0,6:])/4

    # numer of preceding entries (tbparper stands for xpp type xpp branch parameter period)
    # diagram[:,6] is the first state variable over all parameter values
    # diagram[:,:6] are all the xpp types, xpp branches, parameters, periods for all parameter values
    tbparper = 6

    # column index of xpp branch type
    typeidx = 0
    
    # column index of xpp branch number
    bridx = 1
    
    # column index of 0 guy
    zeroidx = 2

    # column index of bifurcation parameters
    par1idx = 3
    par2idx = 4

    # set up array values for retreival
    c1 = []
    c2 = []

    a = np.array([1])
    
    
    c1.append(typeidx)
    c1.append(bridx)
    
    c2.append(par1idx)
    c2.append(par2idx)
    
    for i in range(varnum):
        c2.append(tbparper+i)
    
    c1 = np.array(c1,dtype=int)
    c2 = np.array(c2,dtype=int)
    
    # store various branches to dictionary
    # this dict is for actual plotting values
    val_dict = {}

    # this dict is for type and xpp branch values
    type_dict = {}
    
    # loop over each coordinate. begin new branch if type, branch change values
    # or if parval, period, sv1, sv2, .. svn change discontinuously.
    # first set of comparisons is called c1
    # second set of comparisons is called c2
    
    brnum = 0
    

    val_dict['br'+str(brnum)] = np.zeros((1,2+varnum)) # branches are named in order they are created
    type_dict['br'+str(brnum)] = np.zeros((1,2))


    # initialize
    c1v_prev = np.array([list(diagram[0,c1])])
    c1v = np.array([list(diagram[1,c1])])
    
    c2v_prev = np.array([list(diagram[0,c2])])
    c2v = np.array([list(diagram[1,c2])])


    # val_dict has entries [par1, par2, sv1hi, sv1lo, ..., svnhi, svnlo]
    # type_dict has entries [type, br]
    # for a given xpp branch, consecutive terms are appended as a new row
    val_dict['br'+str(brnum)] = c2v_prev
    type_dict['br'+str(brnum)] = c1v_prev


    for i in range(2,len(diagram[:,0])):
        
        # get values for type and branch
        c1v_prev = np.array([list(diagram[i-1,c1])])
        c1v = np.array([list(diagram[i,c1])])

        # get values for svs and parameters
        c2v_prev = np.array([list(diagram[i-1,c2])])
        c2v = np.array([list(diagram[i,c2])])

        # append above values to current branch
        val_dict['br'+str(brnum)] = np.append(val_dict['br'+str(brnum)],c2v_prev,axis=0)
        type_dict['br'+str(brnum)] = np.append(type_dict['br'+str(brnum)],c1v_prev,axis=0)

        #print type_dict['br'+str(brnum)]

        # if either above threshold, start new key.
        if np.any( np.abs((c1v - c1v_prev))>=1):
            brnum += 1
            
            val_dict['br'+str(brnum)] = c2v
            type_dict['br'+str(brnum)] = c1v
            
        elif np.any( np.abs((c2v - c2v_prev))>=sv_tol):
            brnum += 1
            val_dict['br'+str(brnum)] = c2v
            type_dict['br'+str(brnum)] = c1v


    print val_dict.keys()

    # remove isolated points
    if remove_isolated:
        keyvals = val_dict.keys()
        
        for i in range(len(keyvals)):
            if len(val_dict[keyvals[i]]) <= isolated_number:
                val_dict.pop(keyvals[i])
                type_dict.pop(keyvals[i])


    
    # remove redundant branches
    # a python branch is removed if it shares multiple points (N) with another xpp branch.
    if remove_redundant:


        val_dict_final = {}
        type_dict_final = {}


        # get all xpp branch numbers
        brlist = np.unique(diagram[:,1])

        # collect all branches for each xpp branch number

        keyvals = val_dict.keys()


        keyignorelist = []
        keysavelist = []

        # loop over keys of python branches
        for i in range(len(keyvals)):

            key = keyvals[i]

            if not(key in keyignorelist):

                # get xpp branch number
                xppbrnum = type_dict[key][0,1]

                # loop over remaining python branches
                for j in range(i,len(keyvals)):
                    key2 = keyvals[j]

                    # make sure branches to be compared are distict
                    if not(key2 in keyignorelist) and (key2 != key):

                        #print xppbrnum,key2,type_dict[key2][0,1]

                        # if only 1 xpp branch...
                        

                        # if more than 1 xpp branch
                        # make sure xpp branches are different

                        #if xppbrnum != type_dict[key2][0,1]:
                        if True:



                            # loop over N different values
                            #N = 20
                            belowthresholdcount = 0
                            
                            dN = int(1.*len(val_dict[key][:,0])/N)
                            for i in range(N):
                                # check if N points in val_dict[key] are in val_dict[key2]

                                # first point
                                par1diff = np.amin(np.abs(val_dict[key][dN*i,0]-val_dict[key2][:,0]))
                                par2diff = np.amin(np.abs(val_dict[key][dN*i,1]-val_dict[key2][:,1]))
                                sv1diff = np.amin(np.abs(val_dict[key][dN*i,2]-val_dict[key2][:,2]))
                                sv2diff = np.amin(np.abs(val_dict[key][dN*i,3]-val_dict[key2][:,3]))

                                diff1 = par1diff + par2diff + sv1diff + sv2diff


                                if key == 'br12' and key2 == 'br51':
                                    print par1diff,par2diff,sv1diff,sv2diff
                                    print key,key2,diff1

                                #if (par1diff <= redundant_threshold) or\
                                #   (par2diff <= redundant_threshold) or\
                                #   (sv1diff <= redundant_threshold) or\
                                #   (sv2diff <= redundant_threshold):
                                if diff1 <= redundant_threshold:
                                    #print diff1,key,key2,belowthresholdcount,keyignorelist
                                    #print 'delete', key2
                                    belowthresholdcount += 1
                                    
                            if belowthresholdcount >= 3:

                                keyignorelist.append(key2)
                                #print 'del', key2
                            else:
                                
                                if not(key2 in keysavelist):
                                    #print 'keep', key2
                                    val_dict_final[key2] = val_dict[key2]
                                    type_dict_final[key2] = type_dict[key2]
                                    keysavelist.append(key2)

        for key in keyignorelist:
            if key in keysavelist:

                val_dict_final.pop(key)
                type_dict_final.pop(key)



    else:
        val_dict_final = val_dict
        type_dict_final = type_dict




    if fix_reverse and remove_isolated:
        for key in val_dict_final.keys():
            if val_dict_final[key][2,0] - val_dict_final[key][1,0] < 0:
                for i in range(varnum):
                    val_dict_final[key][:,i] = val_dict_final[key][:,i][::-1]



    return val_dict_final, type_dict_final




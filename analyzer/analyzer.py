import sys
sys.path.insert(0, '../ELINA/python_interface/')


import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time

from analyzer_util import *

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')


def analyze_box(nn, LB_N0, UB_N0, label):
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer
    man = elina_box_manager_alloc()   # manager
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i]) # interval of interest

#construct the box to calculate the range
    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv) # initial abstraction
    elina_interval_array_free(itv,num_pixels)
    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter]
           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           num_out_pixels = len(weights)
           dimadd = elina_dimchange_alloc(0,num_out_pixels) # new dims
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False) # add new dimensions to the box
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels
           # handle affine layer
           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem) # remove the dimensions
           elina_dimchange_free(dimrem)
           # handle ReLU layer
           if(nn.layertypes[layerno]=='ReLU'):
              element = relu_box_layerwise(man,True,element,0, num_out_pixels)
           nn.ffn_counter+=1

        else:
           print(' net type not supported')

    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)


    # if epsilon is zero, try to classify else verify robustness
# check the labels in the calculated range
    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]):
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True
            for j in range(output_size):
                if(j!=i):
                   sup = bounds[j].contents.sup.contents.val.dbl
                   if(inf<=sup):
                      flag = False
                      break
            if(flag):
                predicted_label = i
                break
    else:
        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):
            if(j!=label):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)
    return predicted_label, verified_flag


def main():
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)
    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)

    # get the label
    label, _ = analyze_box(nn,LB_N0,UB_N0,0)
    start = time.time()
    if(label==int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        # this is the key function:
        _, verified_flag = analyze_box(nn,LB_N0,UB_N0,label)
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")


if __name__ == '__main__':
    main()

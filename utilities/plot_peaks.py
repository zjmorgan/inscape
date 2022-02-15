# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

directory = '/home/zgf/Documents/data/Mn3Si2Te6'

fields = [1, 5]
files = ['Mn3Si2Te6_{}T_005K_correction.hkl'.format(field) for field in fields]

reference = 'Mn3Si2Te6_0T_100K_correction.hkl'

datas = [np.loadtxt(os.path.join(directory, file)) for file in files] 

dictionary = {}

for field, data in zip(fields, datas):

    for line in data:
        h, k, l, I, sig, d = line

        key = (int(h),int(k),int(l))

        if dictionary.get(key) is None:
            dictionary[key] = [[I],[field]]
        else:
            items = dictionary[key]
            intens_list, field_list = items
            intens_list.append(I)
            field_list.append(field)
            items = intens_list, field_list
            dictionary[key] = items
            
data = np.loadtxt(os.path.join(directory, reference))

reference_dictionary = {}

for line in data:
    h, k, l, I, sig, d = line

    key = (int(h),int(k),int(l))

    if reference_dictionary.get(key) is None:
        reference_dictionary[key] = [I]
    else:
        intens_list = reference_dictionary[key]
        intens_list.append(I)
        reference_dictionary[key] = intens_list

# for key in reference_dictionary.keys():
# 
#     ref_intens = np.mean(reference_dictionary[key])
# 
#     if dictionary.get(key) is not None:
#         items = dictionary[key]
#         intens_list, field_list = items
#         intens_list = [np.round(intens-ref_intens,2) for intens in intens_list]
#         items = intens_list, field_list
#         dictionary[key] = items
# 
print('\npeak | intensity | field (T)')

peaks = [(1,0,0), (0,1,0), (-1,1,0), (-1,0,0), (0,-1,0), (1,-1,0)]

for peak in peaks:
    print(peak, '|', dictionary[peak][0], '|', dictionary[peak][1])
    
print('\npeak | intensity | field (T)')

peaks = [(1,0,1), (0,1,1), (-1,1,1), (-1,0,1), (0,-1,1), (1,-1,1)]

for peak in peaks:
    print(peak, '|', dictionary[peak][0], '|', dictionary[peak][1])
    
print('\npeak | intensity | field (T)')

peaks = [(1,0,-1), (0,1,-1), (-1,1,-1), (-1,0,-1), (0,-1,-1), (1,-1,-1)]

for peak in peaks:
    print(peak, '|', dictionary[peak][0], '|', dictionary[peak][1])
# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.geometry import PointGroupFactory, SpaceGroupFactory
from mantid.kernel import V3D

directory = '/SNS/CORELLI/IPTS-28994/shared/ErFeO3_CCR_small_tube_test/'
sys.path.append('/SNS/users/zgf/.git/inscape/integration/')

filename = 'ErFeO3_125K_5x7_2p5_10_w_bin_cntr_3d_w_abs_w_ext.hkl'

data = np.loadtxt(os.path.join(directory, filename))

sg = SpaceGroupFactory.createSpaceGroup('P b n m')
pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

miss, total = 0, 0

lines = []

print('Space group #{} ({})'.format(sg.getNumber(),sg.getHMSymbol()))
print('Reflections:')

dictionary = {}

for line in data:

    h, k, l, I, sig, d = line

    hkl = V3D(h,k,l)

    if sg.isAllowedReflection(hkl):

        lines.append(line)

        equivalents = pg.getEquivalents(hkl)

        key = tuple(equivalents)

        if dictionary.get(key) is None:

            dictionary[key] = [len(equivalents),d,[(h,k,l)],[I],[sig]]

        else:

            item = dictionary[key]

            redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item

            intens_list.append(I)
            sig_intens_list.append(sig)
            peak_list.append((h,k,l))

            item = [redundancy, d_spacing, peak_list, intens_list, sig_intens_list]

            dictionary[key] = item

    else:

        print('({},{},{}) forbidden d = {:2.4f} \u212B'.format(int(h),int(k),int(l),d))

        miss += 1

    total += 1

print('{}/{} reflections not allowed in space group'.format(miss,total))

miss, total = 0, 0

data = []

for key in dictionary.keys():

    item = dictionary[key]

    redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item

    intens_list, sig_intens_list = np.array(intens_list), np.array(sig_intens_list)

    median = np.median(intens_list)
    Q1, Q3 = np.percentile(intens_list, [25,75])
    IQR = Q3-Q1

    high = np.argwhere(Q3+1.5*IQR < intens_list).flatten()
    low = np.argwhere(Q1-1.5*IQR > intens_list).flatten()

    if len(high) > 0:
        print('Outlier, intensity too high :', [tuple(peak_list[ind]) for ind in high])

    if len(low) > 0:
        print('Outlier, intensity too low :', [tuple(peak_list[ind]) for ind in low])

    mask = np.concatenate((low,high)).tolist()

    median = np.median(sig_intens_list)
    Q1, Q3 = np.percentile(sig_intens_list, [25,75])
    IQR = Q3-Q1

    high = np.argwhere(Q3+1.5*IQR < sig_intens_list).flatten().tolist()

    mask += high

    for i in range(len(intens_list)):

        if i not in mask:

            h, k, l = peak_list[i]
            I = intens_list[i]
            sig = sig_intens_list[i]
            d = d_spacing

            line = h, k, l, I, sig, d

            data.append(line)

        else:

            miss += 1

        total += 1

print('{}/{} reflections outliers'.format(miss,total))

dictionary = {}

for line in data:

    h, k, l, I, sig, d = line

    hkl = V3D(h,k,l)

    if sg.isAllowedReflection(hkl):

        lines.append(line)

        equivalents = pg.getEquivalents(hkl)

        key = tuple(equivalents)

        if dictionary.get(key) is None:

            dictionary[key] = [len(equivalents),d,[(h,k,l)],[I],[sig]]

        else:

            item = dictionary[key]

            redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item

            intens_list.append(I)
            sig_intens_list.append(sig)
            peak_list.append((h,k,l))

            item = [redundancy, d_spacing, peak_list, intens_list, sig_intens_list]

            dictionary[key] = item

r, n, d = [], [], []

I_sum, I_mae = [], []

for key in dictionary.keys():

    item = dictionary[key]

    redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item
    
    I_mean = np.mean(intens_list)

    r.append(redundancy)
    n.append(np.size(peak_list))
    d.append(d_spacing)

    I_sum.append(np.sum(intens_list))
    I_mae.append(np.sum(np.abs(np.array(intens_list)-I_mean)))

r, n, d = np.array(r), np.array(n), np.array(d)

I_sum, I_mae = np.array(I_sum), np.array(I_mae)

sort = np.argsort(d)[::-1]

r, n, d = r[sort], n[sort], d[sort]
I_sum, I_mae = I_sum[sort], I_mae[sort]

split = np.array_split(np.arange(len(d)), 20)

for s in split:

    d_min, d_max = d[s].min(), d[s].max()

    comp = 100*r[s].sum()/n[s].sum()

    R_merge = 100*I_mae[s].sum()/I_sum[s].sum()
    R_pim = 100*(np.sqrt(1/(n[s]-1))*I_mae[s]).sum()/I_sum[s].sum()

    print('{:2.4f}-{:2.4f} \u212B : Completeness = {:2.2f}%, R(merge) = {:2.2f}%, R(pim) = {:2.2f}%'.format(d_max,d_min,comp,R_merge,R_pim))

hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

new_filename = filename.replace('.hkl', '_prune.hkl')

with open(os.path.join(directory, new_filename), 'w') as f:

    for line in data:

        f.write(hkl_format.format(*line))
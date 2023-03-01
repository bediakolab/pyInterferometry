
import glob
import pandas as pd
import os
import re

def parse_logfile(file, pdict):
    with open(file) as f: lines = f.readlines()

    # basics
    job = file.split('job')[1]
    job = job.split('log')[0]
    pdict["job purpose"] = '?'
    pdict["job number"] = job.strip().strip('\n')
    pdict["path"] = lines[0].strip().strip('\n')
    pdict["date"] = re.findall('[A-Z][a-z]+\s\d{1,2}\s+\d{4}', lines[1])[0].strip().strip('\n')
    pdict["exe"] = lines[2].split('=')[1].strip().strip('\n')
    
    try: 
        pdict["energy"] = lines[3].split('=')[1].strip().strip('\n')
        n = 4
    except:
        n = 3
        pdict["energy"] = 'failed?'

    # psuedopots, outcar
    n = 4
    psuedopots = []
    while 'TITEL' in lines[n]:
        psuedopots.append(lines[n].split('=')[1].strip().strip('\n'))
        n+= 1 
    pdict["psuedopots"] = ' '.join(psuedopots)

    # kpoints

    if 'Automatically generated mesh' in lines[n]:
        pdict["kpoints"] = 'auto mesh'
        pdict["job purpose"] = 'BS'
        n += 6
    elif 'K-points' in lines[n]:
        pdict["kpoints"] = '{} {}'.format(lines[n+2].strip().strip('\n'), lines[n+3].strip().strip('\n'))
        n += 5

    # structure, poscar
    pdict["lattice a1"] = lines[n+2].strip().strip('\n')
    pdict["lattice a2"] = lines[n+3].strip().strip('\n')
    pdict["lattice a3"] = lines[n+4].strip().strip('\n')
    elements = [e for e in lines[n+5].strip().strip('\n').split(' ') if e.strip() != '']
    factors = [e for e in lines[n+6].strip().strip('\n').split(' ') if e.strip() != '']
    pdict["formula"] = ''.join(['{}{} '.format(el, f) for el,f in zip(elements, factors)])
    SD = False
    for i in range(8):
        if 'selective' in lines[n+i].lower():
            SD = True
    pdict["selective dynamics"] = SD
    n += 8

    # inputs, incar
    for i in range(n, len(lines)):
        if '=' in lines[i] and lines[i].strip()[0] != '#': #not a comment or spacer line
            param = lines[i].split('=')[0].strip().strip('\n')
            value = lines[i].split('=')[1].strip().strip('\n').split('#')[0]
            pdict[param] = value

    #for k in pdict.keys():
    #    print('{} --------------- {}'.format(k, pdict[k]))
    #exit()

    return pdict

def main(d):
    path = os.path.join(d, '*log.txt')
    files = glob.glob(path)
    pdict = dict()
    for i in range(len(files)):
        pdict = parse_logfile(files[i] , pdict)
        if i == 0:
            df = pd.DataFrame(data=pdict, index=[0])
            #print(df)
            #exit()
        else: 
            df2 = pd.DataFrame(data=pdict, index=[0])
            #print(df2)
            df = pd.concat([df, df2])
    if len(files) > 0: df.to_excel('summary.xlsx')


main('/Users/isaaccraig/Desktop/')

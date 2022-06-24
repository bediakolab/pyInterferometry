
from utils import bomb_out

# from new_utils
def dump_matrix(mat, savepath):
    mat = np.matrix(mat)
    with open(savepath,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f,')

# from utils - write a file
def writefile(filenm, data_name, data):
    with open(filenm, 'w') as f:
        f.write('{}\n'.format(data_name))
        if isinstance(data,dict):
            for k, v in data.items():
                f.write('{} : {}\n'.format(k, v))
        else:
            for el in data:
                f.write('{}\n'.format(el))

# from utils
def read_excel(fpath):
    data = pd.read_excel(fpath, engine='openpyxl') # openpyxl needed for xlsx files
    return np.matrix(data)

# from new_utils
def parse_filepath(path):
    tmp        = path.replace('\\','/').replace('//','/').split('/')
    tmp        = [el for el in tmp if len(el.strip()) > 0]
    prefix     = tmp[-2]
    dsnum      = int(tmp[-1].split("_")[0])
    tmp        = tmp[-1].split("_")[1]
    scan_shape = [ int(tmp.split("x")[0]), int(tmp.split("x")[1]) ]
    return prefix, dsnum, scan_shape

# from new_utils
def get_diskset_index_options():
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl'))
    return [i for i in range(len(pickles))]

# from new_utils
def import_diskset(indx=None):
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl'))
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    tmp = filepath.replace('\\','/').replace('//','/').split('/')
    tmp = [el for el in tmp if len(el.strip()) > 0]
    prefix = tmp[-2]
    dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
    with open(filepath, 'rb') as f: diskset = pickle.load(f)
    return diskset, prefix, dsnum

# from new_utils
def import_unwrap_uvector(indx=None):
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl_unwrap'))
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    tmp = filepath.replace('\\','/').replace('//','/').split('/')
    tmp = [el for el in tmp if len(el.strip()) > 0]
    prefix = tmp[-2]
    dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
    with open(filepath, 'rb') as f: d = pickle.load(f)
    u, centers, adjacency_type = d[0], d[1], d[2]
    return u, prefix, dsnum, centers, adjacency_type 

# from new_utils
def import_disket_uvector(indx=None):
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl_fit*'))
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    try:
        tmp = filepath.replace('\\','/').replace('//','/').split('/')
        tmp = [el for el in tmp if len(el.strip()) > 0]
        prefix = tmp[-2]
        dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            diskset = data[0]
            if len(data) == 2: uvecs = data[1]
            elif len(data) == 4: coefs, uvecs = data[1:3], data[3]
            elif len(data) == 5: coefs, uvecs = data[1:4], data[4]
        return uvecs, prefix, dsnum, coefs, diskset 
    except: bomb_out('failed reading {}'.format(filepath))

# from new_utils
def import_uvector(indx=None):
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl_fit*'))
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    try:
        tmp = filepath.replace('\\','/').replace('//','/').split('/')
        tmp = [el for el in tmp if len(el.strip()) > 0]
        prefix = tmp[-2]
        dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            diskset = data[0]
            if   len(data) == 2: uvecs = data[1]
            elif len(data) == 4: A, B, uvecs = data[1], data[2], data[3]
            elif len(data) == 5: A, B, C, uvecs = data[1], data[2], data[3], data[4]
        return uvecs, prefix, dsnum
    except: bomb_out('failed reading {}'.format(filepath))

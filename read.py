from functions import read_xfers

src = 'BNL-ATLAS'
dst = 'MWT2'
#filepath = 'data/BNL_MWT2_20180313.csv'
#filepath = 'data/BNL_MWT2_20180405.csv'
filepath = 'data/BNL_MWT2_20180618.csv'
#filepath = 'data/BNL_MWT2_20180619.csv'
#filepath = 'data/BNL_MWT2_20180515.csv'
#filepath = 'data/BNL_MWT2_20180522.csv' # failed due to high R²
#filepath = 'data/BNL_MWT2_20180429.csv'
def readxfers():
    _, xfers = read_xfers(filepath, src, dst)
    return xfers

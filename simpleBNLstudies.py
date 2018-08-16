src = 'BNL-ATLAS'
dst = 'MWT2'

files = [
    'BNL_MWT2_20180213.csv',
    'BNL_MWT2_20180221.csv',
    'BNL_MWT2_20180226.csv',
    'BNL_MWT2_20180313.csv',
    'BNL_MWT2_20180316.csv',
    'BNL_MWT2_20180330.csv',
    'BNL_MWT2_20180405.csv',
    'BNL_MWT2_20180414.csv',
    'BNL_MWT2_20180429.csv',
    'BNL_MWT2_20180515.csv',
    'BNL_MWT2_20180522.csv',
    'BNL_MWT2_20180604.csv',
    'BNL_MWT2_20180618.csv',
    'BNL_MWT2_20180619.csv',
    'BNL_MWT2_20180624.csv',
]

dates = [
    '2018-02-13',
    '2018-02-21',
    '2018-02-26',
    '2018-03-13',
    '2018-03-16'
    '2018-03-30',
    '2018-04-05',
    '2018-04-14',
    '2018-04-29',
    '2018-05-15',
    '2018-05-22',
    '2018-06-04',
    '2018-06-18',
    '2018-06-19',
    '2018-06-24',
]
print('SRC --> DST  | DATE       |  COUNT | MEAN | STD |') 
for f, d in zip(files, dates):
    x, l = read_xfers('data/' + f, src, dst)
    count =  len(l)
    mean = l.QTIME.mean()
    std  = l.QTIME.std()
    print ('BNL --> MWT2 |', d, '|', count, '|', '%05.2f'%mean, '|', '%5.2f'%std, '|')
    x, l = read_xfers('data/' + f, dst, src)
    count =  len(l)
    mean = l.QTIME.mean()
    std  = l.QTIME.std()
    print ('MWT2 --> BNL |', d, '|', count, '|', '%05.2f'%mean, '|', '%5.2f'%std, '|')
    x, l = read_xfers('data/' + f, 'CERN-PROD', 'BNL-ATLAS')
    count =  len(l)
    mean = l.QTIME.mean()
    std  = l.QTIME.std()
    print ('CERN --> BNL |', d, '|', count, '|', '%05.2f'%mean, '|', '%5.2f'%std, '|')
    x, l = read_xfers('data/' + f, 'BNL-ATLAS', 'CERN-PROD')
    count =  len(l)
    mean = l.QTIME.mean()
    std  = l.QTIME.std()
    print ('BNL --> CERN |', d, '|', count, '|', '%05.2f'%mean, '|', '%5.2f'%std, '|')
    print ('')

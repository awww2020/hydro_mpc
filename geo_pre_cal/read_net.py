from geo_pre_cal.geo import Net, Bra, Section, Sec, Po
#打开文件读取
# f = open("断面数据.txt", 'r')
f = open("geo_pre_cal/1_looped_case.txt", 'r')
lines = f.readlines()   # 读取文件中的所有行
print(lines)
dataset = [[] for i in range( len(lines) )]

for i in range(len(dataset)):  # 循环遍历每一行数据
        dataset[i][:] = (item for item in lines[i].split())  # 将每一行字符串按空格分割成一列表，并赋值
        dataset[i][:] = list( map(float,dataset[i][:]) )  # 将列表中的每个元素转换为浮点数
f.close()

# 初始化河网
Net = Net()

# 加载河网数据
l=0 # 初始化一个索引l，用于逐行读取dataset中的数据
Net.np = int(dataset[l][0])  #加[0],把数据从list中读取出来
l +=1
Net.nb = int(dataset[l][0])
l +=1

# 初始化河段
Net.init_Bra()
for i in range(Net.nb):
    # 读取河道数据
    # m,n = dataset[2]
    m,n = dataset[l]
    l += 1
    m = int(m)
    n = int(n)
    ns  = int(dataset[l][0])
    l += 1
    Net.Bra[i] = Bra(m,n,ns)

    # 初始化微段
    Net.Bra[i].init_Sec()
    for j in range((ns+1)):
        nb_point = int(dataset[l][0])
        l += 1
        b_cl, b_cr, nc = dataset[l]
        l += 1
        b_cl = int(b_cl)
        b_cr = int(b_cr)
        b_fl, b_fr, nf = dataset[l]
        l += 1
        b_fl = int(b_fl)
        b_fr = int(b_fr)

        Net.Bra[i].Sec[j] = Sec( nb_point, b_cl,b_cr,nc,b_fl,b_fr,nf)

        Net.Bra[i].Sec[j].init_Po()
        for k in range(nb_point):
            px, py = dataset[l]
            l += 1
            Net.Bra[i].Sec[j].Po[k] = Po(px, py)
        if (j < ns):
            Net.Bra[i].Sec[j].delx = dataset[l]
            l += 1

        Net.Bra[i].Sec[j].zref_cal()

        # for k in range(nb_point):
        #    Net.Bra[i].Sec[j].Po[k].py = Net.Bra[i].Sec[j].Po[k].py - Net.Bra[i].Sec[j].zref

        Net.Bra[i].Sec[j].lay_pre()
        print("finished sec",j+1)

    print("finished bra", i+1)

    X=[0,1]
    Z=[1811,1810]

    # 计算某断面水位下的A和K=1/n*A*R**(2./3)

    '''
    aw=['' for i in range(len(X))]
    kw=['' for i in range(len(X))]
    for i in range(len(X)):
        aw[i], kw[i] = Net.Bra[0].Sec[X[i]].pro_cal(544)
    print(aw,kw)
    '''




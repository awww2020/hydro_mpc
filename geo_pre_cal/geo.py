import math

class Net:
    def __init__(self):
        self.np =[]
        self.nb =[]

        self.ZP = []
        self.Bra = []
        self.A = []  # 方程系数矩阵
        self.C = []  # 常数项

    def init_Bra(self):
        self.Bra = ['' for i in range(self.nb)]

    def solve_time_step(self, dt, boundaries):
        """
        求解单个时间步
        :param dt: 时间步长
        :param boundaries: 边界条件字典
        """
        # 构建全局矩阵
        A = lil_matrix((self.np, self.np))
        b = np.zeros(self.np)

        # 施加边界条件
        for node_id, value in boundaries.items():
            if node_id.endswith('_Q'):
                pass  # 流量边界
            if node_id.endswith('_Z'):
                pass  # 水位边界

        # 求解稀疏矩阵
        A = A.tocsr()
        z = spsolve(A, b)

        return z

class Bra:
    def __init__(self, m, n, ns):
        self.m  = m
        self.n  = n
        self.ns = ns

    def init_Section(self):
        self.Section = ['' for i in range((self.ns))]

    def init_Sec(self):
        self.Sec = ['' for i in range((self.ns+1))]

    def build_global_equation(self, dt):

        for i in range(self.ns):
            self.Section(i).build_equation(dt)


        at1 = self.Section(1).alp1
        bt1 = self.Section(1).bet1
        ct1 = self.Section(1).gam1
        at2 = self.Section(1).alp2
        bt2 = self.Section(1).bet2
        ct2 = self.Section(1).gam2

        for i in range(2,self.ns):
            at01 = -at1 * self.Section(i).alp1 - bt1 * self.Section(i).alp2
            bt01 = -at1 * self.Section(i).bet1 - bt1 * self.Section(i).bet2
            ct01 = -at1 * self.Section(i).gam1 - bt1 * self.Section(i).gam2 + ct1
            at02 = -at2 * self.Section(i).alp1 - bt2 * self.Section(i).alp2
            bt02 = -at2 * self.Section(i).bet1 - bt2 * self.Section(i).bet2
            ct02 = -at2 * self.Section(i).gam1 - bt2 * self.Section(i).gam2 + ct2
            at1 = at01
            bt1 = bt01
            ct1 = ct01
            at2 = at02
            bt2 = bt02
            ct2 = ct02

        # 转化为流量的显格式
        self.aa1 = bt2 / bt1
        self.bb1 = self.aa1 * at1 - at2
        self.cc1 = -self.aa1 * ct1 + ct2
        self.aa2 = -1. / bt1
        self.bb2 = self.aa2 * at1
        self.cc2 = -self.aa2 * ct1



class Section:
    def __init__(self, delx):
        self.delx = delx          # 微段长度
        self.up = Sec()  # 上游断面
        self.dn = None  # 下游断面
        self.alp1 = 0.0
        self.bet1 = 0.0
        self.gam1 = 0.0


    def build_equation(self, dt):

        B1, A1, U1, D1, DPDZ1 = self.up.Bw, self.up.Sw, self.up.U1, self.up.D1, self.up.DPDZ1
        B2, A2, U2, D2, DPDZ2 = self.dn.Bw, self.dn.Sw, self.dn.U1, self.dn.D1, self.dn.DPDZ1
        Z1, Q1 = self.up.Z, self.up.Q
        Z2, Q2 = self.dn.Z, self.dn.Q

        g0 = 9.81
        theta = 0.75

        as1 = ((B2 + B1) * self.delx / (4 * dt * theta))
        bs1 = -1
        cs1 = ((B2 + B1) * self.delx / (4 * dt * theta))
        ds1 = 1
        es1 = -(1 - theta) / theta * (Q2 - Q1) + ((B2 + B1) * self.delx / (4 * dt * theta) * (Z2 + Z1))
        as2 = -g0 * (A1 + A2) / 2
        bs2 = self.delx / (2 * theta * dt) - U1 + (g0 * U2 / (2 * theta * (D1 ** 2) / (A1 ** 2))) * self.delx
        cs2 = g0 * (A1 + A2) / 2
        ds2 = self.delx / (2 * theta * dt) + U2 + (g0 * U2 / (2 * theta * (D2 ** 2) / (A2 ** 2))) * self.delx
        es2 = self.delx * (Q2 + Q1) / (2 * theta * dt) - ((1 - theta) / theta) * (U2 * Q2 - U1 * Q1) - (
                    (1 - theta) / theta) * g0 * (A1 + A2) * (Z2 - Z1) / 2

        Temp = as1 * bs2 - as2 * bs1
        self.alp1 = (cs1 * bs2 - cs2 * bs1) / Temp
        self.bet1 = (ds1 * bs2 - ds2 * bs1) / Temp
        self.gam1 = (es1 * bs2 - es2 * bs1) / Temp
        Temp = -Temp
        self.alp2 = (cs1 * as2 - cs2 * as1) / Temp
        self.bet2 = (ds1 * as2 - ds2 * as1) / Temp
        self.gam2 = (es1 * as2 - es2 * as1) / Temp

class Sec:
    def __init__(self, nb_point, b_cl,b_cr,nc,b_fl,b_fr,nf):
        self.nb_point = nb_point
        self.b_cl = b_cl
        self.b_cr = b_cr
        self.nc = nc
        self.b_fl = b_fl
        self.b_fr = b_fr
        self.nf = nf

        self.Q = 0.0
        self.Z = 0.0
        self.U = 0.0

    def init_Po(self):
        self.Po = ['' for i in range(self.nb_point)]

    def init_Lay(self):
        self.npas = 400   #总层数
        self.pas = 0.01    #垂向层间距
        self.Lay = ['' for i in range(self.npas)]

    def zref_cal(self):
        self.xcl = self.Po[(self.b_cl)-1].px
        self.xcr = self.Po[(self.b_cr)-1].px
        self.xfl = self.Po[(self.b_fl)-1].px
        self.xfr = self.Po[(self.b_fr)-1].px

        # 主槽最低点纵坐标值
        self.zref = self.Po[0].py
        for ipoint in range(1,(self.nb_point-1)):
            self.zref = min( self.zref, self.Po[ipoint].py )

        #左滩地最低点纵坐标值
        self.zref_l = 1.e+9
        for ipoint in range( (self.b_fl - 1), (self.b_cl - 2) ):
            if ( self.Po[ipoint].py <= self.zref_l ):
                self.zref_l = self.Po[ipoint].py

        #右滩地最低点纵坐标值
        self.zref_r = 1.e+9
        for ipoint in range( self.b_cr, (self.b_fr-1) ):
            if ( self.Po[ipoint].py <= self.zref_r):
                self.zref_r = self.Po[ipoint].py

    def lay_pre(self):
        self.init_Lay()
        for ipas in range(self.npas):
            cote = self.zref + ipas * self.pas

            if ipas == 0:
                cote = self.zref + self.pas * 1.e-5
            if abs( cote - self.zref_l) < 1.e-2:
                cote = self.zref_l + self.pas * 1.e-5
            if abs( cote - self.zref_r) < 1.e-2:
                cote = self.zref_r + self.pas * 1.e-5

            #print('lay[i].cote:    ', cote)

            eps8 = 1.e-8
            nb_max_chenaux = 100 #最大河槽数量
            cote_over_max = False
            lg = [0 for i in range(nb_max_chenaux)]
            ld = [0 for i in range(nb_max_chenaux)]

            xg = [0.0 for i in range(nb_max_chenaux)]
            xd = [0.0 for i in range(nb_max_chenaux)]

            # 主槽B / S / P
            b1, s1, p1 = eps8, eps8, eps8

            # 滩地B / S / P
            b2, b2l, b2r = eps8, eps8, eps8
            s2, s2l, s2r = eps8, eps8, eps8
            p2, p2l, p2r = eps8, eps8, eps8

            ipoint, ichenal = 0, 0
            cote_0 = cote
            if (cote > self.Po[ipoint].py):
                cote0 = cote
                cote_0 = self.Po[0].py
                cote_over_max = True
            #print('lay[i].cote_0:  ', cote_0)

            while ipoint < (self.nb_point-1):
                ipoint = ipoint + 1
                flag = False

                if cote_0 > self.Po[ipoint].py:
                    lg[ichenal] = ipoint - 1
                    xg[ichenal] = (self.Po[(ipoint - 1)].px * (self.Po[ipoint].py - cote_0) - self.Po[ipoint].px * (self.Po[(ipoint - 1)].py - cote_0)) \
                                  /(self.Po[ipoint].py - self.Po[(ipoint - 1)].py)
                    while True:
                        if ipoint == (self.nb_point-1):
                            ld[ichenal] = ipoint - 1
                            xd[ichenal] = (self.Po[(ipoint - 1)].px * (self.Po[ipoint].py - cote_0) - self.Po[ipoint].px * (self.Po[(ipoint - 1)].py - cote_0)) \
                                        /(self.Po[ipoint].py - self.Po[(ipoint - 1)].py)
                            ichenal = ichenal + 1

                        if ipoint >= (self.nb_point-1):
                            flag = True
                            break
                        ipoint = ipoint + 1
                        if (cote_0 <= self.Po[ipoint].py):
                            break

                    if flag == True:
                        break

                    ld[ichenal] = ipoint - 1
                    xd[ichenal] = (self.Po[(ipoint - 1)].px * (self.Po[ipoint].py - cote_0) - self.Po[ipoint].px * (
                                self.Po[(ipoint - 1)].py - cote_0)) \
                                  / (self.Po[ipoint].py - self.Po[(ipoint - 1)].py)
                    ichenal = ichenal + 1

            # 计算 B/S/P
            nb_chenaux = ichenal
            #print('nb_chenaux:', nb_chenaux)
            if nb_chenaux > 0:
                for ichenal in range(nb_chenaux):
                    mg = lg[ichenal]
                    xi = self.Po[mg+1].px
                    yi = self.Po[mg+1].py
                    xgj = xg[ichenal]

                    if ( min(xgj, xi) >= self.xcl and max( xgj, xi) <= self.xcr ) :
                        p1 = p1 + math.sqrt((xi - xg[ichenal]) ** 2 + (cote_0 - yi) ** 2)
                        s1 = s1 + 0.5 * (xi - xg[ichenal]) * (cote_0 - yi)
                        b1 = b1 + xi - xg[ichenal]
                    elif( xg[ichenal] >= self.xfl and xi <= self.xcl ) :
                        p2l = p2l + math.sqrt((xi - xg[ichenal]) ** 2 + (cote_0 - yi) ** 2)
                        s2l = s2l + 0.5 * (xi - xg[ichenal]) * (cote_0 - yi)
                        b2l = b2l + xi - xg[ichenal]
                    elif( xg[ichenal] >= self.xcr and xi <= self.xfr ) :
                        p2r = p2r + math.sqrt((xi - xg[ichenal]) ** 2 + (cote_0 - yi) ** 2)
                        s2r = s2r + 0.5 * (xi - xg[ichenal]) * (cote_0 - yi)
                        b2r = b2r + xi - xg[ichenal]

                    # 右岸
                    md = ld[ichenal]
                    xi = self.Po[md].px
                    yi = self.Po[md].py
                    xdj = xd[ichenal]
                    if (min(xi, xdj) >= self.xcl and max( xi, xdj) <= self.xcr ):
                        p1 = p1 + math.sqrt((xd[ichenal] - xi) ** 2 + (cote_0 - yi) ** 2)
                        s1 = s1 + 0.5 * (xd[ichenal] - xi) * (cote_0 - yi)
                        b1 = b1 + xd[ichenal] - xi
                    elif (xi >= self.xfl and xd[ichenal] <= self.xcl ):
                        p2l = p2l + math.sqrt((xd[ichenal] - xi) ** 2 + (cote_0 - yi) ** 2)
                        s2l = s2l + 0.5 * (xd[ichenal] - xi) * (cote_0 - yi)
                        b2l = b2l + xd[ichenal] - xi
                    elif (xi >= self.xcr and xd[ichenal] <= self.xfr ):
                        p2r = p2r + math.sqrt((xd[ichenal] - xi) ** 2 + (cote_0 - yi) ** 2)
                        s2r = s2r + 0.5 * (xd[ichenal] - xi) * (cote_0 - yi)
                        b2r = b2r + xd[ichenal] - xi

                    # 中心
                    largeur_chenal = md - mg

                    #print('largeur_chenal:',largeur_chenal)
                    if largeur_chenal > 0:
                        for ipoint in range(1, largeur_chenal):
                            xig = self.Po[(mg + ipoint)].px
                            yig = self.Po[(mg + ipoint)].py
                            xid = self.Po[(mg + ipoint +1)].px
                            yid = self.Po[(mg + ipoint +1)].py
                            if (min(xig, xid) >= self.xcl and max(xid, xig) <= self.xcr ):
                                p1 = p1 + math.sqrt((xig - xid) ** 2 + (yig - yid) ** 2)
                                s1 = s1 + 0.5 * (xid - xig) * (2.0 * cote_0 - yig - yid)
                                b1 = b1 + xid - xig
                            elif(xig >= self.xfl and xid <= self.xcl):
                                p2l = p2l + math.sqrt((xig - xid) ** 2 + (yig - yid) ** 2)
                                s2l = s2l + 0.5 * (xid - xig) * (2.0 * cote_0 - yig - yid)
                                b2l = b2l + xid - xig
                            elif(xig >= self.xcr and xid <= self.xfr ):
                                p2r = p2r + math.sqrt((xig - xid) ** 2 + (yig - yid) ** 2)
                                s2r = s2r + 0.5 * (xid - xig) * (2.0 * cote_0 - yig - yid)
                                b2r = b2r + xid - xig


            if cote_over_max:
                delta_cote = cote0 - cote_0
                s1 = s1 + b1 * delta_cote
                s2l = s2l + b2l * delta_cote
                s2r = s2r + b2r * delta_cote
                p2l = p2l + delta_cote
                p2r = p2r + delta_cote

            bt = b1 + b2l + b2r
            st = s1 + s2l + s2r
            pt = p1 + p2l + p2r

            k = s1 * (s1/ p1) ** (2. / 3) / self.nc + s2l * (s2l / p2l) ** (2. / 3) / self.nf + s2r * (
                        s2r / p2r) ** (2. / 3) / self.nf

            self.Lay[ipas] = Lay(cote, b1, p1, s1, b2, b2l, b2r, p2, p2l, p2r, s2, s2r, s2l, bt, st, pt, k)

    def pro_cal(self, z):
        sw = 0
        kw = 0
        for i in range( (self.npas-1)):
            cote1 = self.zref + i * self.pas
            cote2 = self.zref + (i + 1) * self.pas
            if ( z>=cote1 and z<cote2 ):

                delta = z - cote1
                '''
                print('z', self.Lay[i+1].cote)
                print('s', self.Lay[i+1].st)
                print('z', self.Lay[i+2].cote)
                print('s', self.Lay[i+2].st)
                '''
                sw = self.Lay[i].st + delta / self.pas * (self.Lay[i+1].st - self.Lay[i].st)  # 断面面积
                # pw = self.Lay[i].pt + delta / pas * (self.Lay[i+1].pt - self.Lay[i+1].pt)  # 断面湿周
                # kw = sw * (sw / pw) ** (2.0 / 3) / self.nc  糙率一样采用这个公式
                self.U = self.Q/sw
                # 主槽 滩地糙率不一样的时候用
                s1w  = self.Lay[i].s1 + delta  / self.pas * (self.Lay[i+1].s1 - self.Lay[i].s1)
                s2lw = self.Lay[i].s2l + delta / self.pas * (self.Lay[i+1].s2l - self.Lay[i].s2l)
                s2rw = self.Lay[i].s2r + delta / self.pas * (self.Lay[i+1].s2r - self.Lay[i].s2r)

                p1w = self.Lay[i].p1 + delta / self.pas * (self.Lay[i+1].p1 - self.Lay[i].p1)
                p2lw = self.Lay[i].p2l + delta / self.pas * (self.Lay[i+1].p2l - self.Lay[i].p2l)
                p2rw = self.Lay[i].p2r + delta / self.pas * (self.Lay[i+1].p2r - self.Lay[i].p2r)

                kw = s1w*(s1w/p1w)**(2./3)/self.nc + s2lw*(s2lw/p2lw)**(2./3)/self.nf + s2rw*(s2rw/p2rw)**(2./3)/self.nf

        return sw, kw


class Po:
    def __init__(self,px,py):
        self.px = px
        self.py = py

class Lay:
    def __init__(self,cote,b1,p1,s1,b2,b2l,b2r,p2,p2l,p2r,s2,s2r,s2l,bt,st,pt,k):
        self.cote = cote
        self.b1 = b1
        self.p1 = p1
        self.s1 = s1
        self.b2 = b2
        self.b2l = b2l
        self.b2r = b2r
        self.p2 = p2
        self.p2l = p2l
        self.p2r = p2r
        self.s2 = s2
        self.s2l = s2l
        self.s2r = s2r
        self.bt = bt
        self.pt = pt
        self.st = st
        self.k = k
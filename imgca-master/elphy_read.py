# -*- coding: utf-8 -*-
# ONLY WORKING WITH PYTHON3
from struct import unpack, calcsize
import numpy as np
from datetime import timedelta, datetime
import math
#import matplotlib.pyplot as plt

def gcd(x, y):
    while y:
        x, y = y, x%y
    return x

def lcm(a,b):
    a= int(a)
    b=int(b)
    return abs(a * b) / gcd(a,b) if a and b else 0


def fread(fichier, repeat, bintype):
    ret = []
    s = calcsize(bintype)
    for i in np.arange(repeat):
        val = unpack(bintype, fichier.read(s))
        if type(val) == tuple:
            val = val[0]
        ret.append(val)
    if repeat==1:
        ret = ret[0]
    return ret

def Nformat(N):
    if N == 0:
        Nf ='B'
    elif N == 1:
        Nf ='b'
    elif N == 2:
        Nf ='h'
    elif N == 3:
        Nf ='H'
    elif N == 4:
        Nf ='i'
    elif N == 5:
        Nf ='f'
    elif N == 7:
        Nf ='d'
    elif N == 9:
        Nf ='f'
    elif N == 10:
        Nf ='d'
    else:
        Nf = ''
    return Nf


def SkipHeader(f):
    l = fread(f,1,'b')
    ID = fread(f,l,'s')
    dum = fread(f,15-l,'b')
    SZ = fread(f,1,'h')
    return


def getBlockID(f):
    if f.read(1) == b'': # WARNING HERE BECAUSE OF READ
        SZ = -1
        return 0,SZ,0
    f.seek(-1,1)
    SZ = fread(f,1,'i')
    l = fread(f,1,'b')
    ID = fread(f,l,'s'); ID = b''.join(ID)
    SZ -= l+5
    posend = f.tell() + SZ
    return ID, SZ, posend

def subBlockID(f):
    l = fread(f,1,'b')
    ID = fread(f, l, 's'); ID = b''.join(ID)
    SZ = fread(f,1,'H')
    if SZ == 65535:
        SZ = fread(f,1,'i')
    posend = f.tell() + SZ
    return ID, SZ, posend


def readString(f,N=0):
    l = fread(f,1,'b');
    if N !=0:
        l = np.minimum(l,N)
    x = fread(f, l ,'s')
    x = b''.join(x)
    if N !=0:
        f.seek(N-l,1)
    return x

def findBlock(f,Name):
    ID, SZ, N = 0, 0, 0
    p = f.tell()
    while (N == 0) and not (f.read(1) == b''): # WARNING HERE BECAUSE OF READ
        if p != f.tell():
            f.seek(-1,1)

        SZ = fread(f, 1, 'i')
        if len([SZ])==0:
            break

        l = fread(f, 1, 'b')
        if len([l]) == 0:
            break

        ID = fread(f,l,'s'); ID = b''.join(ID)
        if ID == Name:
            N += 1
        SZ -= l+5

        if N==0:
            f.seek(SZ,1)

    if (N == 0):
        SZ = -1
        print("Block not found")


    posend = f.tell() + SZ
    return SZ, posend

def GetAgSampleCount(KS):
    N=len(KS)
    ppcm0 = 1
    for  i in np.arange(N):
        if KS[i]>0:
            ppcm0 = lcm(ppcm0,KS[i])
    x=0
    for i in np.arange(N):
        if KS[i]>0:
            x += ppcm0/KS[i]
    return x, ppcm0

def GetMask(KS):
    AgC, _ = GetAgSampleCount(KS)
    Nvoie = len(KS)
    x= np.zeros(int(AgC))
    i=0
    k=0
    while (k+1)<=AgC:
        for j in np.arange(Nvoie):
            if (KS[j]>0) and (i%KS[j] == 0):
                x[k]=j+1   # +1 is a Change to fit to matlab because of python indexing
                k+=1
                if k>AgC:
                    break
            i+=1

    return x

def GetSamplePerChan(nb,AgSz,ppcm0,KS,Ktype,chanMask):
    tpSize = [11,22,44,6,8,10,8,16,20,0]
    x = np.zeros(len(KS))
    nvoie = len(KS)
    nbAg = np.floor(nb/AgSz)

    for i in np.arange(nvoie):
        if KS[i]>0:
            x[i] = nbAg*ppcm0/KS[i]
        else:
            x[i] = 0

    rest = nb%AgSz
    it=0
    j=0
    while it<rest:
        vv = chanMask(j)
        if (vv>0) and (vv<=nvoie):
            x[vv] += 1
            it += tpSize[Ktype[vv]]
            j+=1
    return x

def ReadEpisode(f,SZ,PosMax):
    while f.tell()<PosMax:
        ID, SZ, pos1 = subBlockID(f)
        if ID == b'Ep':
            nbvoie = fread(f,1,'b')
            nbpt = fread(f,1,'i')
            tpData = fread(f,1,'b')
            uX = readString(f,10)
            Dxu = fread(f,1,'d')
            x0u = fread(f,1,'d')
            continu = fread(f,1,'b')
            TagMode = fread(f,1,'b')
            TagShift = fread(f,1,'b')

            if SZ>36:
                DxuSpk = fread(f,1,'d')
                X0uSpk = fread(f,1,'d')
                nbSpk = fread(f,1,'i')
                DyuSpk = fread(f,1,'d')
                Y0uSpk = fread(f,1,'d')
                unitXspk = readString(f,10)
                unitYSpk = readString(f,10)
                CyberTime = fread(f,1,'d')
                PCtime = fread(f,1,'i')

            Ktype = np.ones(nbvoie) * 2

        elif ID == b'Adc': # not tested
            uY = []
            Dyu = []
            Y0u = []
            for ii in np.arange(nbvoie):
                uY.append(readString(f,10))
                Dyu.append(fread(f,1,'d'))
                Y0u.append(fread(f,1,'d'))

        elif ID == b'Ksamp':
            Ksamp = []
            for ii in np.arange(nbvoie):
                Ksamp.append(fread(f,1,'H'))

        elif ID == b'Ktype':
            Ktype = []
            for ii in np.arange(nbvoie):
                Ktype.append(fread(f,1,'b'))

        f.seek(pos1)
    f.seek(PosMax)

    #Read data ('RDATA' block)
    SZ ,_ = findBlock(f,b'RDATA')
    if SZ == -1:
        raise NameError('no episode data')
    RdataHsize = fread(f,1,'h')
    curpos = f.tell()
    f.read(1)
    x = fread(f,1,'Q')
    x = (x*2**-37)-33286456
    date = datetime.fromordinal(int(x)) + timedelta(days=x%1) - timedelta(days = 366)# No idea why ??? To test
    f.seek(curpos+RdataHsize-2)
    SZ -= RdataHsize
    if SZ == 0:
        print("encountered empty recording")
        V= []
        return V, date

    nbSamp = np.floor(SZ/2)
    AgSampleCount, ppcm0 = GetAgSampleCount(Ksamp)
    ChanMask = GetMask(Ksamp)
    SamplePerChan = GetSamplePerChan(nbSamp, AgSampleCount, ppcm0, Ksamp, Ktype, ChanMask)

    N = SamplePerChan
    Dy0 = Dyu
    Y00 = Y0u
    if TagMode == 1:
        Dy0 /= 2**TagShift
    Ktype = np.array(Ktype)

    if type(Ktype==5)==bool:  # modif to fit the np.diff([Ktype==5]) when Ktype is a single value
        Ktype = np.array([Ktype])

    if np.diff([Ktype==5]).any():
        # different types of numbers, go the slow way
        # BEGIN OLD CODE
        Dy0[Ktype == 5] = 1
        if nbvoie > 1:
            raise NameError('not implemented yet')
        V = np.zeros(N)
        k=0
        for i in np.arange(nbSamp):
            im = (i%AgSampleCount)
            if Ktype[ChanMask[im]]==5:
                w = fread(f,1,'f')
            else:
                w = fread(f,1,'h')
            if ChanMask[im] == NumChan:
                V[k] = w*Dy0 + Y00
                k+=1
        # END OLD CODE
    else:
        # go the fast way, load everything at once
        if Ktype[0]==5:
            w = fread(f, int(nbSamp),'f')
        else:
            w = fread(f, int(nbSamp),'h')
        w = np.reshape(w, (int(nbSamp/AgSampleCount),int(AgSampleCount)))
        V = [[]]*nbvoie
        ChanMask -= 1
        for i in np.arange(nbvoie):
            V[i] = np.ravel(w[:,ChanMask == i]*Dy0[i] + Y00[i])

    # TODO:  test to know if all channels have the same size !! No need for the moment so I didn't code it
    return V, date

def ReadVector(f,SZ,PosMax):
    name=[]
    while f.tell()<PosMax:
        ID, SZ, pos1 = subBlockID(f)
        if ID == b'IDENT1':
           name = fread(f,SZ,'s'); name = b''.join(name)
        elif ID == b'OBJINF':
            tpNum = fread(f,1,'b')
            imin = fread(f,1,'i')
            imax = fread(f,1,'i')
            jmin = fread(f,1,'i') # not used for Tvector
            jmax = fread(f,1,'i') # not used for Tvector
            x0u = fread(f,1,'d')
            dxu = fread(f,1,'d')
            y0u = fread(f,1,'d')
            dyu = fread(f,1,'d')
        f.seek(pos1)
    f.seek(PosMax)
    SZ, _ = findBlock(f,b'DATA')
    if SZ == -1:
        raise NameError('no data for vector')
    f.seek(1,1)
    format = Nformat(tpNum)
    V = fread(f, imax-imin+1, format)
    V = np.array(V)*dyu + y0u
    return name, V

def ReadParameters(f,SZ,PosMax):
    # Read header (from sub-blocks of the 'Vector' block)
    while f.tell()<PosMax:
        ID, SZ, pos1 = subBlockID(f)
        if ID == b'ST':
            stri = fread(f,SZ,'s'); stri = b''.join(stri).decode("utf-8")
            names = stri.split("\r\n")[:-1]
            npar = len(names)
        elif ID == b'BUF':
            values =[]
            for i in np.arange(npar):
                typee = fread(f,1,'b')

                if typee == 1:
                    values.append(fread(f, 1, '?'))
                elif typee == 2:
                    values.append(fread(f,1,'q'))
                elif typee == 3:
                    x = fread(f,1,'Q')*2**-63
                    s_e = fread(f,1,'h')
                    s = np.sign(s_e)
                    e = abs(s_e) - (2**14 - 1)
                    values.append(np.sign(s_e)*x*2**e)
                elif typee == 4:
                    l = fread(f,1,'I')
                    if l:
                        st = fread(f,l,'s'); st = b''.join(st).decode("utf-8")
                        st.replace("\\\\","\\")
                        values.append(st)
                    else:
                        values.append('')
                elif typee == 7:
                    l = fread(f,1,"b")
                    st = fread(f,l,'s'); st = b''.join(st).decode("utf-8")
                    st.replace("\\\\","\\")
                    values.append(st)
                else:
                    raise NameError("Problem")
        f.seek(pos1)
    f.seek(PosMax)
    parameters = {names[i]:values[i] for i in np.arange(npar)}
    return parameters

def ReadMemo(f,SZ,PosMax):
    xpar={}
    while f.tell()<PosMax:
        ID, SZ, pos1 = subBlockID(f)
        if ID ==b'IDENT1':
            nameID = fread(f,SZ,'s'); nameID = b''.join(nameID)
        elif ID == b'ST':
            memo = fread(f,SZ,'s'); memo = b''.join(memo).decode('utf-8')
        f.seek(pos1)
    f.seek(PosMax)
    if nameID == b'PG0.PPAR2':
        structure = {}
        isunique = {}
        lines = memo.split("\r\n")[:-1]
        for i in np.arange(len(lines)):
            items = lines[i].split(";")
            name = items[0]
            isunique[name] = (len(items)<3) or len(items[2])==0
            if isunique[name]:
                if len(items)<2:
                    values = ''
                else:
                    values = items[1]
            else:
                values = items[1:]
            for j in np.arange(len(values)):
                if values[j].lower()=='false':
                    values[j]= False
                elif values[j].lower()=='true':
                    values[j]= True
                else:
                    try:
                        values[j] = float(values[j])
                    except:
                        None
            structure[name] = values
        names = np.array(list(structure.keys()))
        values = [structure[nameii] for nameii in names]
        isunique = np.array(list(isunique.values()))
        if isunique.all():
            xpar = {'fix': {names[i]:values[i] for i in np.arange(np.arange(len(names))[isunique])},
                    'table': {}}
        else:
            xpar = {'fix': {names[i]:values[i] for i in np.arange(len(names))[isunique]},
                    'table': {names[i]:values[i] for i in np.arange(len(names))[~isunique]}}
    elif nameID == b'PG0.IMAGELIST':
        xpar = {'imagelist' : memo.split('\r\n')[:-1]}
    elif nameID == b'PG0.SOUNDLIST':
        xpar = {'soundlist' : memo.split('\r\n')[:-1]}
    else:
        print("don't know how to print Memo", nameID.decode("utf-8"))
        xpar[nameID.decode("utf-8")] = memo
    return xpar


def Read(f):
    recordings = []
    vectors = {}
    menupar = []
    xpar = {}
    epinfo = []
    dates=[]

    SkipHeader(f)
    krec=0
    while True:
        ID, SZ, posend = getBlockID(f)
        if SZ == -1:
            break

        if ID == b'B_Ep':
            krec += 1
            a, b = ReadEpisode(f,SZ,posend)
            recordings.append(a)
            dates.append(b)
        elif ID == b'Vector':
            name, value = ReadVector(f,SZ,posend)
            name = name.decode("utf-8").lower()
            name = name.replace('pg0.', '')
            vectors[name] = value
            f.seek(posend)
        elif ID == b'DBrecord':
            parameters=ReadParameters(f,SZ,posend)
            if 'ProtocolType' in parameters:
                if len(menupar) == 0:
                    menupar = parameters
                else:
                    print("problem ! several set of parameters in file")
            else:
                 epinfo.append(parameters)
        elif ID == b'Memo':
            xpar.update(ReadMemo(f,SZ,posend))
        else:
            if ID != b'DATA':
                None

        f.seek(posend)
    return np.transpose(recordings,(0,2,1)), dates, vectors, menupar, xpar, epinfo



# f=open("1649F_CXG2.DAT",'rb')
# recordings, dates, vectors, menupar, xpar, epinfo = Read(f)

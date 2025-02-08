
from utils.hparameter import *
import torch
import numpy as np
import crcmod
import bchlib
import random

def add_crc(wm):
    full_wm = str(wm.numpy().tolist())[1:-1].replace('.0','').replace(',','').replace(' ','')
    a = bytes(full_wm, encoding='utf-8')
    CRC = crcmod.predefined.Crc(CRC_MODULE)
    CRC.update(a)
    a = CRC.crcValue
    a = bin(a)
    a = str(a)[2:]
    padding = CRC_LENGTH-len(a)
    for i in range(padding):
        a = '0'+a
    crc = torch.Tensor([int(i) for i in a])
    return torch.cat([wm,crc],dim=0)



def verify_crc(wm):
    full_wm = str(wm.numpy().tolist())[1:-1].replace('.0','').replace(',','').replace(' ','')
    wm = full_wm[:-CRC_LENGTH]
    crc = full_wm[-CRC_LENGTH:]
    a = bytes(wm, encoding='utf-8')
    CRC = crcmod.predefined.Crc(CRC_MODULE)
    CRC.update(a)
    a = CRC.crcValue
    if a == int(crc,2):
        try:
            return bytes(int(wm[i : i + 8], 2) for i in range(0, len(wm), 8)).decode('utf-8')
        except Exception as e:
            pass
    else:
        return False



def add_bch(wm):
    full_wm = str(wm.numpy().tolist())[1:-1].replace('.0','').replace(',','').replace(' ','')
    print(f"{len(full_wm)}:{full_wm}")
    a = int(full_wm,2)
    a = bin(a)
    print(f"{len(a)}:{a}")
    a = bytearray(a, encoding='utf-8')
    print(f"{len(a)}:{a}")
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    # data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(a)
    print(f"{len(ecc)}:{ecc}")
    ecc = int.from_bytes(ecc, "big")
    ecc = bin(ecc)
    print(f"{len(ecc)}:{ecc}")
    ecc = bytearray(ecc, encoding='utf-8') # bytearray(b'0b101011010011110101010101011011011100000')
    print(f"{len(ecc)}:{ecc}")
    a = str(a)[14:-2]
    ecc = str(ecc)[14:-2]
    return torch.Tensor([int(i) for i in a+ecc])


def new_add_bch(wm):
    full_wm = str(wm.numpy().tolist())[1:-1].replace('.0','').replace(',','').replace(' ','')
    print(f"{len(full_wm)}:{full_wm}")
    full_wm_bytes = bytes(int(full_wm[i : i + 8], 2) for i in range(0, len(full_wm), 8))
    a = bytearray(full_wm_bytes)
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    ecc = bch.encode(a)
    ecc = ''.join(format(x, '08b') for x in ecc)
    print(f"{len(ecc)}:{ecc}")
    return_ = [int(i) for i in full_wm+ecc]
    # return_.extend([1,0,1,0])
    return torch.Tensor(return_)



# do error correct
def do_ec(wm):
    full_wm = "".join([str(int(bit)) for bit in wm[:96]])
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    full_wm = bytes(int(full_wm[i : i + 8], 2) for i in range(0, len(full_wm), 8))
    full_wm= bytearray(full_wm)
    wm = full_wm[:-bch.ecc_bytes]
    ecc = full_wm[-bch.ecc_bytes:]
    bitflips = bch.decode_inplace(wm, ecc)
    if bitflips != -1:
        try:
            code = wm.decode("utf-8")
            print(code)
        except:
            pass
    else:
        pass
    wm = ''.join(format(x, '08b') for x in wm)
    return torch.Tensor([int(i) for i in wm])

def test_crc():
    a = [1 for i in range(16)]
    b = [0 for i in range(16)]
    a.extend(b)
    random.shuffle(a)
    a = np.array(a)
    a = torch.Tensor(a)
    a = add_crc(a)
    print(f"{a.shape}:{a}")
    #a[12:15] = 0.  # make distortion
    results = verify_crc(a)
    print(results)



def test_bch():
    a = [1 for i in range(68)]
    b = [0 for i in range(68)]
    a.extend(b)
    random.shuffle(a)
    a = np.array(a)
    a = torch.Tensor(a)
    # a = torch.cat((a,b),dim=0)
    bched = new_add_bch(a)
    # print(f"{len(bched)}:{bched}")

    # bched[1:20] = 0. # add noise
    for i in range(0):
        if bched[i] == 0.:
            bched[i] = 1.
        else:
            bched[i] = 0.
    
    wm = do_ec(bched)

    wm = str(wm.numpy().tolist())[1:-1].replace('.0','').replace(',','').replace(' ','')
    print(f"{len(wm)}:{wm}")
    return 0


def test_all():
    a = [1 for i in range(16*8)]
    b = [0 for i in range(16*8)]
    a.extend(b)
    random.shuffle(a)
    a = np.array(a)
    a = torch.Tensor(a) 
    a = add_crc_bch(a)
    print(f"bched:{a}")
    if_find = search_crc(a)
    print(f"if find:{if_find}")


def add_crc_bch(wm):
    crced = add_crc(wm)
    bched = new_add_bch(crced)
    return bched

def search_crc(bched):
    ec_crced = do_ec(bched)
    verify_result = verify_crc(ec_crced)
    return verify_result

if __name__ == "__main__":
    # test_crc()
    a = test_bch()
    # test_all()
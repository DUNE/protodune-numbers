import os.path as osp
from collections import namedtuple
import numpy
import csv

WireChannel = namedtuple("WireChannel","iplane,letter,wire,ch,asic")

def convert_shanshan_v3(files):
    '''
    Convert ASIC number swap fixes from Shanshan.

    $ sha1sum ProtoDUNE_APA_Wire_Mapping_091917_v3.xlsx
    37293bf334e9755819b7f2f4b9ffc72211d4b55c  ProtoDUNE_APA_Wire_Mapping_091917_v3.xlsx

    Save CON1 and CON2 as CSV files

    $ sha1sum *.csv
    717b95ccebdeac9c0602457b657a20e4a0227fa1  ProtoDUNE_APA_Wire_Mapping_091917_v3-J1.csv
    d64e1f255ca1468a986f112414811bbf298563bc  ProtoDUNE_APA_Wire_Mapping_091917_v3-J2.csv
    '''
    SSletters="UVX"
    letters="uvw"
    
    ret = list()
    def makeit(cells):
        iplane = SSletters.index(cells[0][0])
        wire = int(cells[0][1:])
        letter = letters[iplane]
        ch = int(cells[7])
        asic = int(cells[6])
        # cib = int(cells[11]) some number 0-127
        wc = WireChannel(iplane, letter, wire, ch, asic)
        ret.append(wc)
        
    for fname in files:
        page = csv.reader(open(fname))
        for line in page:
            if not line[0]:
                continue
            first = line[0]
            if not first[0] in SSletters:
                continue
            makeit(line[:12])
            makeit(line[13:])
    return ret
            

def convert(files):
    ret = list()
    for fname in files:
        for line in open(fname).readlines():
            if line.startswith("j"):
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            letter = parts[1]
            iplane = "uvw".index(letter)
            wire,ch,asic = [int(n) for n in parts[2:]]
            ret.append(WireChannel(iplane,letter,wire,ch,asic))
    return ret


def matrixify(wch):
    '''
    Make a table of ASIC vs channel number holding w#.
    '''
    arr = numpy.ndarray(shape=(8,16), dtype=tuple)   # (plane,wire)
    for one in wch:
        arr[one.asic-1, one.ch] = (one.letter, one.wire)
    return arr
def latexmatrix(m):
    '''
    Return latex text for body of matrix made from m.
    '''
    color = dict(u="red", v="blue", w="black")

    lines = list()
    for chn, ch in enumerate(m.T):
        cells = ["ch%02d" % chn]
        for plane, wire in ch:
            cells.append(r"\textcolor{%s}{%s%02d}" % (color[plane], plane, wire))
        lines.append(" & " .join(cells))
    end = r"\\" + "\n"
    body = end.join(lines)

    top = "&".join(["ASIC:"] + [str(n+1) for n in range(8)]) + r"\\"

    form = "r|rrrrrrrr"

    tabular = [r"\begin{tabular}{%s}"%form, r"\hline", top, r"\hline", body+r"\\", r"\hline", r"\end{tabular}"]
    return "\n".join(tabular)

if '__main__' == __name__:
    import sys
    files = sys.argv[1:]        # femb-channel-map-j[12].csv or ProtoDUNE_APA_Wire_Mapping_091917_v3-J[12].csv
    basename = osp.basename(files[0])
    if basename.startswith("femb-channel-map"):
        wch  = convert(files)
    elif basename.startswith("ProtoDUNE_APA_Wire_Mapping_091917_v3-J"):
        wch  = convert_shanshan_v3(files)
    else:
        raise RuntimeError("I have no idea what these files are: " + basename)
    mch = matrixify(wch)
    lch = latexmatrix(mch)
    print "% generated by chmap.py using files:"
    for fname in files:
        print "% " + fname
    print lch

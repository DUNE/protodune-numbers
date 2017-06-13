from collections import namedtuple
import numpy

WireChannel = namedtuple("WireChannel","iplane,letter,wire,ch,asic")

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

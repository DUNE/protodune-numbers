#!/usr/bin/env python
'''
Process file produced by:

https://cdcvs.fnal.gov/redmine/projects/dunetpc/wiki/_ProtoDUNE-SP_Wire_Dumps_

Columns:

0) channel :: [0-15359]
1) cryostat :: 0
2) tpc :: [0,11]
3) plane :: [0,2]
4) wire :: [0,1147]
5) wire beg x
6) wire beg y
7) wire beg z
8) wire end x
9) wire end y
10) wire end z

'''

import sys
import numpy 
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict

# wid simply counts wire in order it is seen in the file
# wip is wire-in-plane
Wire = namedtuple("Wire", "seg wid ch cryo tpc plane wip beg end")


def load(infile):
    wires = list()

    chan2wires = defaultdict(list)

    with open(infile) as fp:
        for wid, line in enumerate(fp.readlines()):
            chunks = line.strip().split()
            numbers = map(int, chunks[:5]) + map(lambda x: 10.0*float(x), chunks[5:])
            zmax = max(numbers[7], numbers[10])
            wire = (zmax, wid, numbers)
            chan = numbers[0]
            chan2wires[chan].append(wire)

    wires = list()
    for ch,segs in chan2wires.items():
        segs.sort()
        for iseg, seg in enumerate(segs):
            zmax, wid, numbers = seg
            wire = Wire(iseg, wid, numbers[0], numbers[1], numbers[2], numbers[3], numbers[4],
                            numbers[5:8], numbers[8:11])
            wires.append(wire)
    return wires

def select_by(wires, key, val):
    ret = list()
    for wire in wires:
        if getattr(wire, key, None) == val:
            ret.append(wire)
    return ret


def summary(wires):
    numcols = ["ch","cryo","tpc","plane","wip"]
    num = dict()
    for icol, col in enumerate(numcols):
        num[col] = len(set([getattr(w,col) for w in wires]))
        print ("#%s: %d " % (col, num[col]))

    for itpc in range(num["tpc"]):
        for iplane in range(num["plane"]):
            witp = select_by(select_by(wires, "tpc", itpc), "plane", iplane)
            c = set([w.ch for w in witp])
            print ("tpc:%d plane:%d #chans:%d #wires:%d" %
                       (itpc, iplane, len(c), len(witp)))
            
        
    

def plot_wires_beg(wires, nwires, title):
    plt.clf()
    xyz="XYZ"
    axes=(2,1)
    cmap = plt.get_cmap('seismic')
    nwires = len(wires)
    colors = [cmap(i) for i in numpy.linspace(0, 1, nwires)]
    for ind, wire in enumerate(wires):
        if ind%100 != 0:
            continue
        p1,p2 = wire[5:8], wire[8:11]
        x = numpy.asarray((p1[axes[0]], p2[axes[0]]))
        y = numpy.asarray((p1[axes[1]], p2[axes[1]]))
        plt.plot(x, y, color=colors[ind])
        plt.text(0.5*sum(x), 0.5*sum(y), "ch %d" % wire[0])
    plt.xlabel("%s direction" % xyz[axes[0]])
    plt.ylabel("%s direction" % xyz[axes[1]])
    plt.title(title)


def plot_wires_plane_tpc(wires, plane, tpcs=(1,5,9), wire_sampling=10, text_sampling=100, offsets=None):
    plt.clf()
    xyz="XYZ"
    uvw="UVW"
    axes=(2,1)
    #cmap = plt.get_cmap('seismic')
    cmap = plt.get_cmap('rainbow')

    inplane = select_by(wires, "plane", plane)
    for itpc, tpc in enumerate(tpcs):
        offset = numpy.zeros(2)
        if offsets:
            offset = numpy.asarray(offsets[itpc])

        intpc = select_by(inplane, "tpc", tpc)
        nwires = len(intpc)
        colors = [cmap(i) for i in numpy.linspace(0, 1, nwires)]
        todraw = [(i,w) for i,w in enumerate(intpc) if i%wire_sampling == 0]
        todraw.append((nwires-1, intpc[-1]))
        for ind,wire in todraw:
            p1,p2 = wire.beg, wire.end
            x = numpy.asarray((p1[axes[0]], p2[axes[0]]))
            y = numpy.asarray((p1[axes[1]], p2[axes[1]]))
            plt.plot(x, y, color=colors[ind])
            if ind%text_sampling == 0:
                plt.plot([0.5*sum(x)], [0.5*sum(y)], marker='o', color=colors[ind])

                x = numpy.asarray((p1[axes[0]], p2[axes[0]])) + offset[0]
                y = numpy.asarray((p1[axes[1]], p2[axes[1]])) + offset[1]
                plt.text(0.5*sum(x), 0.5*sum(y), 
                             "tpc:%d\nwid:%d\nwip:%d\nch:%d" % (tpc, wire.wid, wire.wip, wire.ch),
                             verticalalignment='center', horizontalalignment='center')
    plt.xlabel("%s direction" % xyz[axes[0]])
    plt.ylabel("%s direction" % xyz[axes[1]])
    plt.title("%s-plane, TPCs: %s" % (uvw[plane], str(tpcs)))
            

def plot_wires(wires):
    print ("Plotting %d wires" % len(wires))

    for wire in wires:
        p1, p2 = wire.beg, wire.end
        x = numpy.asarray((p1[2], p2[2]))
        y = numpy.asarray((p1[1], p2[1]))
        plt.plot(x, y)
        


from matplotlib.backends.backend_pdf import PdfPages

def wire_plots(wires, filename = "larsoftwires.pdf"):
    tses = [100, 100, 170]

    with PdfPages(filename) as pdf:

        plot_wires_plane_tpc(wires, 0)
        pdf.savefig()

        plot_wires_plane_tpc(wires, 1)
        pdf.savefig()

        plot_wires_plane_tpc(wires, 2, offsets=[(0,20), (0,-20), (0,+20)], text_sampling=100)
        pdf.savefig()

        plot_wires_plane_tpc(wires, 0, tpcs=(0,1), offsets=[(8,0), (-8,0)])
        pdf.savefig()

        plot_wires_plane_tpc(wires, 1, tpcs=(0,1), offsets=[(8,0), (-8,0)])
        pdf.savefig()

        plot_wires_plane_tpc(wires, 2, tpcs=(0,1), offsets=[(0,20), (0,-20)], text_sampling=50)
        pdf.savefig()




def do_wire_plots(filename="WireDump_ProtoDUNESP_v2_2.txt", outfile="larsoftwires.pdf"):
    wires = load(filename)
    wire_plots(wires, outfile)


#!/usr/bin/env python
'''
Wires and Channels
'''

import math
from collections import namedtuple

Wire = namedtuple("Wire", "index Chan w1 h1 w2 h2")

class Point(object):
    def __init__(self, *coords):
        self._coords = list(coords)

    def __str__(self):
        s = ",".join([str(a) for a in self._coords])
        return "Point(%s)" % s

    def __repr__(self):
        return str(self)

    @property
    def x(self):
        return self[0]
    @x.setter
    def x(self, val):
        self[0] = val

    @property
    def y(self):
        return self[1]
    @y.setter
    def y(self, val):
        self[1] = val

    def __len__(self):
        return len(self._coords)
    def __getitem__(self, key):
        return self._coords[key]
    def __setitem__(self, key, val):
        self._coords[key] = val
    def __iter__(self):
        return self._coords.__iter__()

    def __abs__(self):
        return Point(*[abs(a) for a in self])

    def __sub__(self, other):
        try:
            return Point(*[(a-b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a-other) for a in self])

    def __add__(self, other):
        try:
            return Point(*[(a+b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a+other) for a in self])

    def __mul__(self, other):
        try:
            return Point(*[(a*b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a*other) for a in self])

    def __div__(self, other):
        try:
            return Point(*[(a/b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a/other) for a in self])

    def dot(self, other):
        return sum([a*b for a,b in zip(self, other)])

    @property
    def magnitude(self):
        return math.sqrt(self.dot(self))

    @property
    def unit(self):
        mag = self.magnitude
        return self/mag

class Ray(object):
    def __init__(self, tail, head):
        self.tail = tail
        self.head = head

    def __str__(self):
        return "%s -> %s" % (self.tail, self.head)

    def __repr__(self):
        return str(self)

    @property
    def vector(self):
        return self.head - self.tail

    @property
    def unit(self):
        return self.vector.unit
    

class Rectangle(object):
    def __init__(self, width, height, center = Point(0.0, 0.0)):
        self.width = width
        self.height = height
        self.center = center

    @property
    def ll(self):
        return Point(self.center.x - 0.5*self.width,
                         self.center.y - 0.5*self.height);

    def relative(self, point):
        return point - self.center

    def inside(self, point):
        r = self.relative(point)
        return abs(r.x) <= 0.5*self.width and abs(r.y) <= 0.5*self.height

    def toedge(self, point, direction):
        '''
        Return a vector that takes point along direction to the nearest edge.
        '''
        p1 = self.relative(point)
        d1 = direction.unit
        
        #print "toedge: p1:%s d1:%s" % (p1, d1)

        corn = Point(0.5*self.width, 0.5*self.height)

        xdir = d1.dot((1.0, 0.0))             # cos(theta_x)
        if xdir == 0:
            tx = None
        else:
            xsign = xdir/abs(xdir)
            dx = xsign*corn.x - p1.x
            tx = dx/d1.x

        ydir = d1.dot((0.0, 1.0))             # cos(theta_y) 
        if ydir == 0:
            ty = None
        else:
            ysign = ydir/abs(ydir)
            dy = ysign*corn.y - p1.y
            ty = dy/d1.y


        if ty is None:
            return d1*tx
        if tx is None:
            return d1*ty

        if tx < ty:            # closer to vertical side
            return d1 * tx
        return d1 * ty



def wrap_one(start_ray, rect):
    '''
    Return wire end points by wrapping around a rectangle.
    '''
    p = rect.relative(start_ray.tail)
    d = start_ray.unit
    ret = [p]
    while True:
        #print "loop: p:%s d:%s" %(p,d)
        jump = rect.toedge(p, d)
        p = p + jump
        ret.append(p)
        if p.y <= -0.5*rect.height:
            break
        d.x = -1.0*d.x                    # swap direction
    return ret
        
    


def wrapped_from_top(offset, angle, pitch, rect):
    '''
    Cover a rectangle with a plane of wires starting along the top of
    the given rectangle and starting at given offset from upper-left
    corner of the rectangle and with angle measured from the vertical
    to the wire direction.  Positive angle means the wire starts going
    down-left from the top of the rectangle.
    '''
    cang = math.cos(angle)
    sang = math.sin(angle)
    direc = Point(-sang, -cang)
    pitchv = Point(cang, -sang)

    start = Point(-0.5*rect.width + offset, 0.5*rect.height) + rect.center

    step = pitch / cang
    stop = rect.center.x + 0.5*rect.width

    print -0.5*rect.width, start.x, step, stop

    wires = list()

    channel = 0
    while True:
        points = wrap_one(Ray(start, start+direc), rect)
        side = 1
        for seg, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
            wcenter = (p1+p2)*0.5 - rect.center
            along_pitch = pitchv.dot(wcenter)
            w = (along_pitch, side, channel, seg, p1, p2)
            wires.append(w)
            side *= -1
        start.x += step
        if start.x >= stop:
            break
        channel += 1
    return wires
        

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy

def plot_rect(rect, color="black"):
    ax = plt.axes()
    ax.add_patch(mpatches.Rectangle(rect.ll, rect.width, rect.height,
                                        color=color, fill=False))
    ax.set_xlabel("APA-local Z")
    ax.set_ylabel("APA-local Y")
    ax.set_title("Looking in anti-drift direction")
    

def plot_polyline(pts):
    cmap = plt.get_cmap('seismic')
    npts = len(pts)
    colors = [cmap(i) for i in numpy.linspace(0, 1, npts)]
    for ind, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
        x = numpy.asarray((p1.x, p2.x))
        y = numpy.asarray((p1.y, p2.y))
        plt.plot(x, y,  linewidth=ind+1)
    

def plotwires(wires):
    cmap = plt.get_cmap('seismic')
    nwires = len(wires)

    chans = [w[2] for w in wires]
    minchan = min(chans)
    maxchan = max(chans)
    nchans = maxchan - minchan + 1

    colors = [cmap(i) for i in numpy.linspace(0, 1, nchans)]
    for ind, one in enumerate(wires):
        pitch, side, ch, seg, p1, p2 = one
        linestyle = 'solid'
        if side < 0:
            linestyle = 'dashed'
        color = colors[ch-minchan]

        x = numpy.asarray((p1.x, p2.x))
        y = numpy.asarray((p1.y, p2.y))
        plt.plot(x, y, color=color, linewidth = seg+1, linestyle = linestyle)

def plot_wires_sparse(wires, indices, group_size=40):
    for ind in indices:
        plotwires([w for w in wires if w[2]%group_size == ind])


def plot_some():
    rect = Rectangle(6.0, 10.0)
    plt.clf()
    direc = Point(1,-1);
    for offset in numpy.linspace(.1, 6, 60):
        start = Point(-3.0 + offset, 5.0)
        ray = Ray(start, start+direc)
        pts = wrap_one(ray, rect)
        plot_polyline(pts)

mm = 1.0
meter = 1000.0*mm
deg = math.pi/180.0

protodune_params = dict(
    active_width = 2295*mm,
    active_height = 5920*mm,
    pitches = [4.669*mm, 4.669*mm, 4.790*mm ],
    # guess at left/right ambiguity
    angles = [+35.707*deg, -35.707*deg,  0.0],
    # guess based on symmetry and above numbers
    offsets = [0.3923*mm, 0.3923*mm, 0.295*mm],
    )

def protodune_plane_one_side(letter="u"):
    iplane = "uvw".index(letter.lower())

    rect = Rectangle(protodune_params['active_width'], protodune_params['active_height'])

    wires =  wrapped_from_top(protodune_params['offsets'][iplane], 
                                  protodune_params['angles'][iplane],
                                  protodune_params['pitches'][iplane],
                                  rect)
    return rect,wires


def celltree_geometry():
    '''
    Spit out contents of a file like:

    https://github.com/BNLIF/wire-cell-celltree/blob/master/geometry/ChannelWireGeometry_v2.txt

    columns of:
    # channel plane wire sx sy sz ex ey ez
    '''
        

    #Wire = namedtuple("Wire", "index Chan w1 h1 w2 h2")
    # wire: (along_pitch, side, channel, seg, p1, p2)
    aps = set()
    sides = set()
    channels = set()
    for iplane, letter in enumerate("uvw"):
        rect, wires = protodune_plane_one_side(letter)
        print letter, len(wires)
        for wire in wires:
            ap, side, channel, seg, p1, p2 = wire
            print wire
            aps.add(ap)
            sides.add(side)
            channels.add(channel)
    print len(aps),len(sides),len(channels)

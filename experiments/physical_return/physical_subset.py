"""Pinned oscillator subset of anttiluode/ChainNeuronsWithPhaseSpike.

This is a NumPy reimplementation of the actual KG field, receiver-pump phase
equation, finite Gaussian monomer kernels, and isolated self-field calibration
in chains/world.py. It is NOT the original World class; it excludes motion,
force calculations, ligation, held_by and bond dynamics because the pairs
are pinned. Phase RNG uses NumPy, not Torch; source-less physics is otherwise
explicit below. No software memory register, homeostat or explicit relays.
"""
import copy
import math
import numpy as np


class Field:
    def __init__(self, pair_centers=((24., 24.),), detune=.015, N=48, seed=0,
                 g_phase=.02, g_emit=.1, noise=0., gamma=.01, m2=.02, c2=.3):
        self.N=N; self.g_phase=g_phase; self.g_emit=g_emit; self.noise=noise
        self.gamma=gamma; self.m2=m2; self.c2=c2
        self.rng=np.random.default_rng(seed); self.t=0
        self.pos=np.array([(x-1.2,y) for x,y in pair_centers]+
                          [(x+1.2,y) for x,y in pair_centers],dtype=float)
        self.npairs=len(pair_centers)
        self.om=np.array([.03]*self.npairs+[.11+detune]*self.npairs)
        self.theta=self.rng.random(len(self.om))*2*np.pi
        self.psi=np.zeros((N,N),float); self.prev=self.psi.copy()
        self.indices=[]; self.weights=[]
        for x,y in self.pos:
            bx,by=np.rint((x,y)).astype(int); c=[]; wg=[]
            for dy in range(-2,3):
                for dx in range(-2,3):
                    px=bx+dx; py=by+dy
                    d2=(px-x)**2+(py-y)**2
                    val=np.exp(-d2/(2*.8**2))*(d2<=2.2**2)
                    c.append((py%N,px%N)); wg.append(val)
            self.indices.append(np.array(c))
            self.weights.append(np.array(wg)/sum(wg))
        self.self_resp=self.calibrate_self()
        self.emit_mask=np.ones(len(self.om),dtype=float)
        self.src_kernels={}

    def lap(self,x):
        return np.roll(x,1,0)+np.roll(x,-1,0)+np.roll(x,1,1)+np.roll(x,-1,1)-4*x

    def calibrate_self(self,steps=3000):
        """Isolated monochromatic kernel field calibrated via cos/sin LSQ."""
        freqs=np.unique(self.om); ans={}
        n=40; offsets=[(dy,dx) for dy in range(-2,3) for dx in range(-2,3)]
        d2=np.array([dy*dy+dx*dx for dy,dx in offsets],float)
        w=np.exp(-d2/(2*.8**2))*(d2<=2.2**2); w/=w.sum()
        ys=np.array([(20+dy)%n for dy,dx in offsets])
        xs=np.array([(20+dx)%n for dy,dx in offsets])
        for freq in freqs:
            f=np.zeros((n,n)); fp=f.copy();theta=0.;xx=[];yy=[]
            for i in range(steps):
                hear=np.sum(f[ys,xs]*w)
                if i>steps//2:
                    xx.append((math.cos(theta),math.sin(theta)));yy.append(hear)
                q=math.cos(theta);theta+=float(freq)
                new=(2-self.gamma)*f-(1-self.gamma)*fp+self.c2*self.lap(f)-self.m2*f
                np.add.at(new,(ys,xs),self.g_emit*q*w)
                fp,f=f,new
            ans[float(freq)]=np.linalg.lstsq(np.array(xx),np.array(yy),rcond=None)[0]
        return np.array([ans[float(x)] for x in self.om])

    def fork(self):
        return copy.deepcopy(self)

    def step(self,source=None):
        c=math.cos(.14*self.t)
        th=self.theta.copy()
        heard=np.array([np.sum(self.psi[idx[:,0],idx[:,1]]*w)
                         for idx,w in zip(self.indices,self.weights)])
        heard-=self.self_resp[:,0]*np.cos(th)+self.self_resp[:,1]*np.sin(th)
        self.theta=(th+self.om-self.g_phase*c*heard*np.sin(th)/self.om+
                    self.rng.normal(0,self.noise,len(th)))%(2*np.pi)
        new=(2-self.gamma)*self.psi-(1-self.gamma)*self.prev+self.c2*self.lap(self.psi)-self.m2*self.psi
        for k,(idx,w) in enumerate(zip(self.indices,self.weights)):
            np.add.at(new,(idx[:,0],idx[:,1]),
                      self.g_emit*self.emit_mask[k]*math.cos(th[k])*w)
        if source is not None:
            center,amp=source
            x,y=center
            if center not in self.src_kernels:
                yy,xx=np.mgrid[:self.N,:self.N]
                dx=(xx-x+self.N/2)%self.N-self.N/2
                dy=(yy-y+self.N/2)%self.N-self.N/2
                gauss=np.exp(-(dx*dx+dy*dy)/2.)
                self.src_kernels[center]=gauss/gauss.sum()
            new+=amp*self.src_kernels[center]
        self.prev,self.psi=self.psi,new
        self.t+=1

    def phases(self):
        return np.angle(np.exp(1j*(self.theta[:self.npairs]-
                                  self.theta[self.npairs:])))

    def sums(self):
        return np.angle(np.exp(1j*(self.theta[:self.npairs]+
                                  self.theta[self.npairs:]-.14*self.t)))

    def observable(self,k):
        return np.sum(self.psi[self.indices[k][:,0],self.indices[k][:,1]]*self.weights[k])

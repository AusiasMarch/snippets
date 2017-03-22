#!/usr/bin/python
import sys,os,math,glob

# Read user provided system description
sys.path.insert(0,'./')
from match_setup import *

def newlist(N):
    x=[]
    for i in xrange(N):
        x.append(0.)
    return x

def scal(X,Y):
    L=len(X)
    Z=[]
    for i in xrange(L):
        Z.append(X[i]*Y[i])
    return Z

def dotprod(vec1,vec2):
    N=len(vec1)
    dp=0.
    for i in xrange(N):
        dp=dp+vec1[i]*vec2[i]
    return dp

try:
    armijo_f=armijo_f
except:
    armijo_f=1e-4


def do_line_search(values,
                   feval,
                   feval_value,
                   maxiter,
                   accuracy,
                   grads,
                   realgrads,
                   newvalues,
                   newgrads,
                   initialstep,
                   private):
    N=len(values)
    x=newlist(3)
    v=newlist(3)
    stepsize=initialstep
    error=0.
    niter=0
    fitreached=0
    olddiff=0.

    armijo_k=armijo_f*dotprod(grads,realgrads)
    
    for j in range(3):
        x[j]=stepsize*j
        for i in xrange(N):
            newvalues[i]=values[i]-grads[i]*x[j]
        v[j],nev=feval_value(newvalues,private)
        niter=niter+nev

    armijo_base=v[0]

    walk=1
    while 1:
        # x/v arrays are always kept sorted
        if (v[1]<v[0] and v[1]<v[2]) or not walk:
            walk=0
            if v[0]>v[2]:
                larg=0
            else:
                larg=2
            try:
            # Fit to quadratic polynomial a+b*x+c*x*x
                c=((-v[0]*x[2]+v[0]*x[1]-v[2]*x[1]+x[2]*v[1]
                    -x[0]*v[1]+x[0]*v[2])/
                   (x[2]*x[1]*x[1]-x[2]*x[2]*x[1]+x[0]*x[2]*x[2]
                    -x[0]*x[1]*x[1]-x[0]*x[0]*x[2]+x[0]*x[0]*x[1]))
                b=(-(x[1]*x[1]*v[0]-v[2]*x[1]*x[1]-x[2]*x[2]*v[0]
                     +v[2]*x[0]*x[0]-v[1]*x[0]*x[0]+v[1]*x[2]*x[2])/
                   ((-x[2]+x[1])*(x[1]*x[2]-x[1]*x[0]+x[0]*x[0]-x[2]*x[0])))
                a=((-v[1]*x[0]*x[0]*x[2]+v[2]*x[0]*x[0]*x[1]
                    -v[2]*x[0]*x[1]*x[1]+v[1]*x[0]*x[2]*x[2]
                    +x[1]*x[1]*v[0]*x[2]-x[2]*x[2]*v[0]*x[1])/
                   (x[2]*x[1]*x[1]-x[2]*x[2]*x[1]+x[0]*x[2]*x[2]
                    -x[0]*x[1]*x[1]-x[0]*x[0]*x[2]+x[0]*x[0]*x[1]))
            except:
                print "FPE terminating linesearch"
                stepsize=0.
                c=0.
            if c<0.:
                print "Quadratic fit found a maximum. Reverting to walk!"
                sys.stdout.flush()
                walk=1
            else:
                if abs(c)<1e-15:
                    print "Linesearch c: too small:",c
                    stepsize=0.
                else:
                    # Obtain minimum in quadratic function
                    xnew=-b/(2*c)
                    print "Linesearch: Quadratic old x=",x[1]," new x=",xnew
                    sys.stdout.flush()
                    for i in xrange(N):
                        newvalues[i]=values[i]-grads[i]*xnew
                    vnew,nev=feval_value(newvalues,private)
                    niter=niter+nev
                    if not fitreached:
                        stepsize=(x[1]-x[0])*0.25
                    #if vnew<v[larg] and xnew>x[0] and xnew<x[2]:
                    if vnew<v[larg]:
                        v[larg]=vnew
                        x[larg]=xnew
                    else:
                        print "Reverting to walk"
                        sys.stdout.flush()
                        walk=1
                        stepsize=stepsize*0.5
                fitreached=1
        if walk:
            # Walk.
            # Take smallest two values and extend.
            # If we walk forward we double the stepsize.
            # If we walk backward we multiply the stepsize with 0.9
            # Pick points.
            sv=newlist(2)
            if v[0]<v[1]:
                sv[0]=0
                smallest=0
            else:
                sv[0]=1
                smallest=1
            if v[2]<v[1-sv[0]]:
                sv[1]=2
                svlast=1-sv[0]
            else:
                sv[1]=1-sv[0]
                svlast=2
            if v[2]<v[smallest]:
                smallest=2
            # Direction
            if abs(x[sv[0]]-x[sv[1]])<1e-15:
                print "Linesearch: x diff too small"
                stepsize=0.
            else:
                dir=(v[sv[0]]-v[sv[1]])/(x[sv[0]]-x[sv[1]])
                if dir<0: # forward
                    if not fitreached:
                        stepsize=stepsize*2
                    else:
                        stepsize=stepsize*0.9
                    stepsign=1
                else: # backward
                    stepsize=stepsize*0.5
                    stepsign=-1
                x[svlast]=x[smallest]+stepsize*stepsign
                print "Linesearch: Walk new x=",x[svlast]
                sys.stdout.flush()
                xnew=x[svlast]
                for i in xrange(N):
                    newvalues[i]=values[i]-grads[i]*xnew
                vnew,nev=feval_value(newvalues,private)
                niter=niter+nev
                if not fitreached:
                    stepsize=(x[1]-x[0])*0.25
                if vnew<v[svlast]:
                    v[svlast]=vnew
                    x[svlast]=xnew
                else:
                    print "Walk halve stepsize"
                    sys.stdout.flush()
                    stepsize=stepsize*0.5

        # sort points
        for i in range(3):
            for j in range(i+1,3):
                if x[i]>x[j]:
                    tmp=x[j]
                    x[j]=x[i]
                    x[i]=tmp
                    tmp=v[j]
                    v[j]=v[i]
                    v[i]=tmp
        print "Sorted points:"
        print x[0],v[0]
        print x[1],v[1]
        print x[2],v[2]
        sys.stdout.flush()

        maxiter=maxiter-1
        diff=abs(v[0]-v[1])+abs(v[2]-v[1])
        error=abs(diff-olddiff)
        olddiff=diff
        print "Line search: error=",error," stepsize=",stepsize
        sys.stdout.flush()

        armijo_cond=0
        for i in range(3):
            armijo=armijo_base+armijo_k*x[i]
            if x[i]>1e-15 and v[i]<armijo:
                print "Armijo condition holds for ",x[i],":",v[i],"<=",armijo,"(",armijo_base,"+",armijo_k,"*",x[i],")"
                armijo_cond=1
        if maxiter<0 or error<accuracy or stepsize<=1e-15 or armijo_cond:
            print "Breaking now!",maxiter,error,stepsize,armijo_cond
            break
    if v[0]<v[1]:
        if v[2]<v[0]:
            x[0]=x[2]
    else:
        if v[2]<v[1]:
            x[0]=x[2]
        else:
            x[0]=x[1]
    for i in xrange(N):
        newvalues[i]=values[i]-grads[i]*x[0]
    v[0],nev=feval(newvalues,newgrads,private)
    niter=niter+nev
    return niter,x[0],0,0

def compute_init_step(v,a):
    a=abs(a)
    inis=a
    N=len(v)
    # Largest move if we use a:
    larg=0.
    for i in xrange(N):
        if abs(v[i]*a)>larg:
            larg=abs(v[i]*a)
    if abs(larg)<1e-15:
        larg=1.
        inis=1e-3
    elif larg>defaultalpha: # Seems too large
        inis=a*defaultalpha/larg
    elif larg<0.001: # Seems too small
        inis=a*0.001/larg
    print "Init step is ",inis," since v=",v," and a=",a
    return inis

def polak_ribiere(values,feval,feval_value,maxiter,accuracy,private):
    N=len(values)
    newvalues=newlist(N)
    newgrads=newlist(N)
    g=newlist(N)
    h=newlist(N)
    xi=newlist(N)
    realgrads=newlist(N)
    largest=0.
    niter=0
    alpha=defaultalpha
    alphafail=defaultalpha
    e,nev=feval(values,xi,private)
    niter=niter+nev
    for i in xrange(N):
        realgrads[i]=h[i]=g[i]=xi[i]
    for i in xrange(N):
        if largest<abs(xi[i]):
            largest=abs(xi[i])
    while 1:
        print "Polak Ribiere: Starting new line search"
        sys.stdout.flush()
        initstep=compute_init_step(xi,defaultalpha)
        nitnew,alpha,fail,usefast=do_line_search(values,feval,
                                                 feval_value,
                                                 maxiter-niter,
                                                 linesearchacc,
                                                 xi,realgrads,newvalues,
                                                 newgrads,initstep,
                                                 private)
        if fail:
            # Reinitialize h, g and xi
            print "Polak Ribiere: Line search failed. Restarting."
            e,nev=feval(values,xi,private)
            niter=niter+nev
            largest=0.
            for i in xrange(N):
                realgrads[i]=h[i]=g[i]=xi[i]
            for i in xrange(N):
                if largest<abs(xi[i]):
                    largest=abs(xi[i])
            # Init new varying alpha here:
            alpha=alphafail
            alphafail=alphafail*0.5
        else:
            niter=niter+nitnew
            print "Polak Ribiere: alpha=",alpha
            sys.stdout.flush()
            largest=0.
            for i in xrange(N):
                realgrads[i]=newgrads[i]
            for i in xrange(N):
                if largest<abs(newgrads[i]):
                    largest=abs(newgrads[i])
            print "Polak Ribiere: Largest gradient is:",largest
            sys.stdout.flush()
            if largest>accuracy:
                dgg=0.
                gg=0.
                for i in xrange(N):
                    xi[i]=-newgrads[i]
                for i in xrange(N):
                    gg=gg+g[i]*g[i]
                    dgg=dgg+(xi[i]+g[i])*xi[i]
                gam=dgg/gg
                for i in xrange(N):
                    g[i]=-xi[i]
                    xi[i]=g[i]+gam*h[i]
                    h[i]=xi[i]
            for i in xrange(N):
                values[i]=newvalues[i]
            if largest<=accuracy or niter>=maxiter:
                break

def steepest_descent(values,feval,feval_value,maxiter,accuracy,private):
    N=len(values)
    newvalues=newlist(N)
    newgrads=newlist(N)
    d=newlist(N)
    largest=0.
    niter=0
    alpha=defaultalpha
    alphafail=defaultalpha
    e,nev=feval(values,d,private)
    niter=niter+nev
    for i in xrange(N):
        if largest<abs(d[i]):
            largest=abs(d[i])
    while 1:
        print "Steepest descent: Starting new line search"
        sys.stdout.flush()
        initstep=compute_init_step(d,defaultalpha)
        nitnew,alpha,fail,usefast=do_line_search(values,feval,
                                                 feval_value,
                                                 maxiter-niter,
                                                 linesearchacc,
                                                 d,d,newvalues,
                                                 newgrads,initstep,
                                                 private)
        niter=niter+nitnew
        print "Steepest descent: alpha=",alpha
        sys.stdout.flush()
        largest=0.
        for i in xrange(N):
            if largest<abs(newgrads[i]):
                largest=abs(newgrads[i])
        print "Steepest descent: Largest gradient is:",largest
        sys.stdout.flush()
        for i in xrange(N):
            values[i]=newvalues[i]
            d[i]=newgrads[i]
        if largest<=accuracy or niter>=maxiter:
            break

def matvec(mat,vecin,vecout):
    N=len(vecin)
    for i in xrange(N):
        tmp=0.
        for j in xrange(N):
            tmp=tmp+mat[N*i+j]*vecin[j]
        vecout[i]=tmp

def vvmat(vec1,vec2,mat):
    N=len(vec1)
    for i in xrange(N):
        for j in xrange(N):
            mat[N*i+j]=vec1[i]*vec2[j]  

def matmat(N,mat1,mat2,matout):
    for i in xrange(N):
        for j in xrange(N):
            tmp=0.
            for k in xrange(N):
                tmp=tmp+mat1[N*i+k]*mat2[N*k+j]
            matout[N*i+j]=tmp

def bfgs(values,feval,feval_value,maxiter,accuracy,private):
    N=len(values)
    hessinv=newlist(N*N)
    gradold=newlist(N)
    grad=newlist(N)
    newvalues=newlist(N)
    newgrad=newlist(N)
    realgrads=newlist(N)
    yk=newlist(N)
    sk=newlist(N)
    sk_skT=newlist(N*N)
    sk_ykT=newlist(N*N)
    hessinv_yk=newlist(N)
    hessinv_yk_skT=newlist(N*N)
    sk_ykT_hessinv=newlist(N*N)
    alpha=defaultalpha
    alphafail=defaultalpha
    failmaxnumberinit=10
    failmaxnumber=failmaxnumberinit
    # Initialize inverse hessian to unit matrix
    for i in xrange(N):
        for j in xrange(N):
            hessinv[N*i+j]=0.
    for i in xrange(N):
        hessinv[N*i+i]=1.
    niter=0
    e,nev=feval(values,grad,private)
    niter=niter+nev
    # Correct gradient sign here
    for i in xrange(N):
        grad[i]=-grad[i]
    largest=0.

    for i in xrange(N):
        if largest<abs(grad[i]):
            largest=abs(grad[i])
    while 1:
        for i in xrange(N):
            realgrads[i]=grad[i]
        # Negative gradient
        for i in xrange(N):
            grad[i]=-grad[i]
        # Line search direction
        matvec(hessinv,grad,sk)
        # Line search
        print "BFGS: Starting new line search"
        sys.stdout.flush()
        initstep=compute_init_step(sk,defaultalpha)
        nitnew,alpha,fail,usefast=do_line_search(values,feval,
                                                 feval_value,
                                                 maxiter-niter,
                                                 linesearchacc,
                                                 sk,realgrads,newvalues,
                                                 newgrad,initstep,
                                                 private)
        #if abs(alpha)<1e-15:
        #    fail=1
        if fail:
            # Reinitialize inverse hessian
            print "BFGS: Line search failed. Restarting with unit matrix hessian"
            failmaxnumber-=1
            if failmaxnumber<=0:
                print "BFGS: Too many restarts. Stopping."
                break
            for i in xrange(N):
                for j in xrange(N):
                    hessinv[N*i+j]=0.
            for i in xrange(N):
                hessinv[N*i+i]=1.
            e,nev=feval(values,grad,private)
            niter=niter+nev
            # Correct gradient sign here
            for i in xrange(N):
                grad[i]=-grad[i]
            largest=0.
            # Init new varying alpha here:
            alpha=alphafail
            alphafail=alphafail*0.5
        else:
            # Allow new restarts
            failmaxnumber=failmaxnumberinit
            # Correct gradient sign here
            for i in xrange(N):
                newgrad[i]=-newgrad[i]
            niter=niter+nitnew
            print "BFGS: alpha=",alpha
            sys.stdout.flush()
            if abs(alpha)<1e-15:
                break
            # Update hessian using the BFGS formula
            ialpha=1./alpha
            for i in xrange(N):
                yk[i]=ialpha*(newgrad[i]+grad[i]) # grad already negative!
            matvec(hessinv,yk,hessinv_yk)
            skT_yk=dotprod(sk,yk)
            ykT_hessinv_yk=dotprod(yk,hessinv_yk)
            vvmat(sk,sk,sk_skT)
            vvmat(sk,yk,sk_ykT)
            vvmat(hessinv_yk,sk,hessinv_yk_skT)
            matmat(N,sk_ykT,hessinv,sk_ykT_hessinv)
            if abs(skT_yk)<1e-15:
                break
            s1=(skT_yk+ykT_hessinv_yk)/(skT_yk**2)
            s2=1./skT_yk
            for i in xrange(N):
                for j in xrange(N):
                    hessinv[N*i+j]=(hessinv[N*i+j]+
                                    s1*sk_skT[N*i+j]-
                                    s2*(hessinv_yk_skT[N*i+j]+
                                        sk_ykT_hessinv[N*i+j]))
    #        for i in xrange(N):
    #            for j in xrange(N):
    #                sys.stdout.write(" "+`hessinv[N*i+j]`)
    #            sys.stdout.write("\n")

            for i in xrange(N):
                grad[i]=newgrad[i]
                values[i]=newvalues[i]

            largest=0.
            for i in xrange(N):
                if largest<abs(grad[i]):
                    largest=abs(grad[i])
            print "BFGS: Largest gradient is:",largest
            sys.stdout.flush()
            if largest<=accuracy or niter>=maxiter:
                break


def replace_parameters(files,params,fitchar):
    parcnt=0
    for fn in files:
        tmpfile=open("tmpfile","w")
        f=open(fn)
        while 1:
            line=f.readline()
            if not line:
                break
            ls=line.split(fitchar)
            if len(ls)!=1:
                line=ls[0]+`params[parcnt]*precond[parcnt]`+ls[2]
                parcnt+=1
            tmpfile.write(line)
        f.close()
        tmpfile.close()
        os.rename("tmpfile",fn)

# Compute the forces on each of the atoms in this config using the mdsim.d program
def evaluate_multiple_configs(params,dumpnums,private,rank):
    os.mkdir("tmpdir_mdeval"+`rank`)
    os.system("cp input.inp obeli.x tmpdir_mdeval"+`rank`)
    os.chdir("tmpdir_mdeval"+`rank`)
    replace_parameters(private[2],params,private[3])
    f=open("eval_for.sh","w")
    f.write("#!/bin/bash \n")
    for dumpnum in dumpnums:
        f.write("cp ../restart.res."+`dumpnum`+" last_conf.out \n")
        f.write("./obeli.x \n")
        f.write("cp forces.out md.forces."+`dumpnum`+"\n")
        if dip_fit:
            f.write("cp mol_mu.out md.dipols."+`dumpnum`+"\n")
#            f.write("cp corr-mol_mu.out  md.dipols."+`dumpnum`+"\n")
    f.close()

    os.chmod("eval_for.sh",0744)
    os.system("./eval_for.sh")
    rforces=[]
    for dumpnum in dumpnums:
        rforce=[]
        f=open("md.forces."+`dumpnum`)
        btoa=0.529177
        fs=2625.4996251/btoa  # Forces in KJ/(mol A)
        while 1:
            line=f.readline()
            if not line:
                break
            data=line.split()
            rforce.append((float(data[0])*fs,float(data[1])*fs,float(data[2])*fs))
        f.close()
        hasforcemapcall=0
        try:
            fmapcall=force_map_sites
            hasforcemapcall=1
        except:
            rforceX=rforce
        if hasforcemapcall:
            rforceX=fmapcall(rforce)
        rforces.append(rforceX)

    if dip_fit:
        rdipols=[]
        for dumpnum in dumpnums:
            rdipol=[]
            f=open("md.dipols."+`dumpnum`)
            fs=1
            while 1:
                line=f.readline()
                if not line:
                    break
                data=line.split()
                rdipol.append((float(data[0])*fs,float(data[1])*fs,float(data[2])*fs))
            f.close()
            hasforcemapcall=0
            try:
                fmapcall=force_map_sites
                hasforcemapcall=1
            except:
                rdipolX=rdipol
            if hasforcemapcall:
                rforceX=fmapcall(rforce)
            rdipols.append(rdipolX)
    
    os.chdir("..")
    os.system("rm -rf tmpdir_mdeval"+`rank`)
    if dip_fit:
        return rforces, rdipols
    else :
        return rforces

def myweight(f):
    try:
        wf=weight_fn(f)
    except:
        wf=1.
    return wf

def calc_multiple_framedata(params,alldata,private,dumpnums,rank):
    val=0.
    et=0.
    ef=0.
    ei=0.
    ed=0.
    valff=0.
    valtt=0.
    valdd=0.
    
    rs=evaluate_multiple_configs(params,dumpnums,private,rank)
    if dip_fit:
        rforces=rs[0]
        rdipols=rs[1]
    else :
        rforces=rs
    
    if scat:
        sf=open("scatter_f.out","w")
        st=open("scatter_t.out","w")
        si=open("scatter_i.out","w")
        sd=open("scatter_d.out","w")
    for kk in xrange(len(dumpnums)):
        k=dumpnums[kk]
        coords=alldata[k][1]
        forces=alldata[k][2]
        weight_f=0.
        weight_t=0.
        weight_i=0.
        weight_d=0.
        valt=0.
        valf=0.
        vali=0.
        vald=0.
        if dip_fit:
            dipols=alldata[k][3]
        for i in xrange(len(forces)):
            acceptit=1
            dis=1
            if acceptit:
                if ((i)%3)==0: # Water
                    # Evaluate center of mass forces, so intramolecular forces are ignored.
                    comf=[0,0,0]
                    comfr=[0,0,0]
                    comw=[16,1,1]

                    if dip_fit:
                        if (abs(rforces[kk][i][0])<0.0000001) and (abs(rforces[kk][i][1])<0.0000001) and (abs(rforces[kk][i][2])<0.00000001):
                            dis=0
                            # print `i`+" discarted"

                    # Intra-molecular force
                    if intra_fit :
                        for k in xrange(3):
                            for j in xrange(3):
                                f=forces[i+k][j]
                                fr=rforces[kk][i+k][j]
                                if w1 :
                                    vali+=dis*(f**2)*(f-fr)**2
                                    weight_i+=dis*(f**4)
                                if w2 :
                                    vali+=dis*(f-fr)**2
                                    weight_i+=dis*(f**2)
                                if w3 :
                                    vali+=dis*abs(f)*(f-fr)**2
                                    weight_i+=dis*abs(f**3)
                                if scat:
                                    si.write(`f`+"  "+`fr`+" \n")
                                if rmse:
                                    ei+=(f-fr)**2
                    d=0.
                    dr=0.
                    if dip_fit :
                        for j in xrange(3):
                            d+=dipols[i/3][j]**2
                            dr+=rdipols[kk][i/3][j]**2
                        d**=0.5
                        dr**=0.5
                        #print `i/3`+"  "+`d`+"  "+`dr`
                        vald+=dis*(abs(d-2.97)**alpha_d)*(d-dr)**2
                        weight_d+=dis*abs(d-2.97)**(alpha_d)*(d-2.97)**2
                        if scat:
                            sd.write(`d`+"  "+`dr`+" \n")
                        if rmse:
                            ed+=(d-dr)**2
                            
                            
                    if f_fit :
                        # Net Force over the molecule
                        for k in xrange(3):
                            for j in xrange(3):
                                comf[j]+=forces[i+k][j]
                                comfr[j]+=rforces[kk][i+k][j]

                        for j in xrange(3):
                            valf+=dis*abs((comf[j])**alpha_f)*(comf[j]-comfr[j])**2
                            weight_f+=dis*abs(comf[j]**(alpha_f+2))
                            if scat:
                                sf.write(`comf[j]`+"  "+`comfr[j]`+" \n")
                            if rmse:
                                ef+=(comf[j]-comfr[j])**2
                    if t_fit :
                        # Compute torque and weight for it
                        com=[0,0,0]
                        torq=[0,0,0]
                        torqr=[0,0,0]
                        racom=[]
                        for k in xrange(3):
                            for j in xrange(3):
                                com[j]+=comw[k]*coords[i+k][j]
                        for k in xrange(3):
                            com[k]/=(comw[0]+comw[1]+comw[2])
                        
                        for k in xrange(3):
                            racom.append([])
                            for j in xrange(3):
                                racom[k].append(coords[i+k][j]-com[j])
                            
                        for k in xrange(3):
                            torq[0]+=forces[i+k][2]*(racom[k][1])-forces[i+k][1]*(racom[k][2])
                            torq[1]+=forces[i+k][0]*(racom[k][2])-forces[i+k][2]*(racom[k][0])
                            torq[2]+=forces[i+k][1]*(racom[k][0])-forces[i+k][0]*(racom[k][1])
                        
                            torqr[0]+=rforces[kk][i+k][2]*(racom[k][1])-rforces[kk][i+k][1]*(racom[k][2])
                            torqr[1]+=rforces[kk][i+k][0]*(racom[k][2])-rforces[kk][i+k][2]*(racom[k][0])
                            torqr[2]+=rforces[kk][i+k][1]*(racom[k][0])-rforces[kk][i+k][0]*(racom[k][1])

                        for j in xrange(3):
                            valt+=dis*abs((torq[j])**alpha_t)*(torq[j]-torqr[j])**2
                            weight_t+=dis*abs(torq[j]**(alpha_t+2))
                            if scat:
                                st.write(`torq[j]`+"  "+`torqr[j]`+" \n")
                            if rmse:
                                et+=(torq[j]-torqr[j])**2
                                
        if f_fit :
            valf/=(weight_f)
            valff+=valf
        if t_fit :
            valt/=(weight_t)
            valtt+=valt
        if intra_fit :
            vali/=(weight_i)
        if dip_fit :
            vald/=(weight_d)
            valdd+=vald
        val+=vali+vald+valf+valt
        #print "discarted molecules on frame = "+`kk`+"  "+`discar`
    print "vals "+`valff/val*100`+" "+`valtt/val*100`+" "+`valdd/val*100`
    if rmse:
        rmsef=(ef/(natoms*len(dumpnums)))**0.5/4.184
        rmset=(et/(natoms*len(dumpnums)))**0.5/4.184
        rmsei=(ei/(natoms*3*len(dumpnums)))**0.5/4.184
        rmsed=(ed/(natoms/3*len(dumpnums)))**0.5
        rm=open("rmse.out","w")
        rm.write("RMSE F = "+`rmsef`+" Kcal/(mol A) \n")
        rm.write("  \n")
        rm.write("RMSE T = "+`rmset`+" Kcal/mol \n")
        rm.write("  \n")        
        rm.write("RMSE I = "+`rmsei`+" Kcal/(mol A)\n")
        rm.write("  \n")        
        rm.write("RMSE D = "+`rmsed`+" D \n")
    if scat:
        sf.close
        st.close
        si.close
        sd.close
    if rmse:
        rm.close
    return val

def parallel_evaluate(params,private,parallel):
    numdata=len(private[1])
    numper=numdata/parallel
    fromto=[]
    for i in xrange(parallel):
        fromto.append([i*numper,(i+1)*numper])
    fromto[parallel-1][1]=numdata
    pidpipes=[]
    for i in xrange(parallel):
        pipe=os.pipe()
        pid=os.fork()
        if pid:
            # Parent
            os.close(pipe[1])
            pidpipes.append((pipe[0],pid))
        else:
            # Worker process
            os.close(pipe[0])
            f=os.fdopen(pipe[1],"w")
            val=0.
            #print "rank",i,"does from",fromto[i][0],"to",fromto[i][1]
            val+=calc_multiple_framedata(params,private[1],private,range(fromto[i][0],fromto[i][1]),i)
            #print "rank",i,"got val",val
            f.write(`val`+"\n")
            f.close()
            os._exit(0)
    # Collect values
    val=0.
    pleft=parallel
    while pleft:
        (pid,status)=os.wait()
        pleft-=1
        for i in xrange(parallel):
            if pidpipes[i][1]==pid:
                f=os.fdopen(pidpipes[i][0],"r")
                val+=float(f.readline())
                f.close()
                try:
                    os.close(pidpipes[i][0])
                except:
                    None
    return val
            

def evaluate_the_value(params,private):
    val=0.
    numproc=1
    try:
        numproc=parallel
    except:
        numproc=1
    
    if numproc!=1:
        val=parallel_evaluate(params,private,parallel)
    else:
        val+=calc_multiple_framedata(params,private[1],private,range(len(private[1])),0)
    print "parameters is ",scal(params,precond), " the value is ",val
    sys.stdout.flush()
    return val

def numgrad(params,private):
    numpar=len(params)
    grad=[]
    nev=0
    for i in xrange(numpar):
        delta=abs(params[i]*gradd)
        newpar=params[:]
        newpar[i]-=delta
        nval=evaluate_the_value(newpar,private)
        nev=nev+1
        newpar=params[:]
        newpar[i]+=delta
        pval=evaluate_the_value(newpar,private)
        grad.append((pval-nval)/(delta*2))
    print "Gradient is",grad
    sys.stdout.flush()
    return grad,nev

def feval(values,grads,private):
    step=private[0]
    step=step+1
    private[0]=step
    e=evaluate_the_value(values,private)
    nev=1
    print "Function evaluation: ",e
    sys.stdout.flush()
    grad,nevnew=numgrad(values,private)
    nev=nev+nevnew
    for i in xrange(len(values)):
        grads[i]=grad[i]
    return e,nev

def feval_value(values,private):
    step=private[0]
    step=step+1
    private[0]=step
    e=evaluate_the_value(values,private)
    print "Function evaluation: ",e
    sys.stdout.flush()
    return e,1

def moveatom(x,L):
    return x-L*math.floor(x/L)

def minimg(x0,x1,L):
    dx=x1-x0
    dx-=L*math.floor(dx/L+0.5)
    return x0+dx



# Check if we can create a directory to do the fit in
try:
    os.stat("fitdir")
    print "fitdir already exists"
    sys.exit(1)
except:
    None

# The same for the result
try:
    os.stat("resultdir")
    print "resultdir already exists"
    sys.exit(1)
except:
    None

# Create a directory to do the fit in
os.mkdir("fitdir")
# Copy all necessary files there
os.system("cp input.inp obeli.x fitdir")
# Go there
os.chdir("fitdir")


# Which character to determine what is a parameter
try:
    fitchar=fit_character
except:
    fitchar=","

# All files which can contain parameters
modelfiles=glob.glob("*.inp")

# Figure out the number of parameters we should change, and their initial values
params=[]
for fn in modelfiles:
    f=open(fn)
    while 1:
        line=f.readline()
        if not line:
            break
        ls=line.split(fitchar)
        if len(ls)!=1:
            if len(ls)==3:
                params.append(float(ls[1]))
            else:
                print "Bad number of fit characters >>"+fitchar+"<< on line: "+line
                sys.exit(1)
    f.close()

try:
    mymethod=optmethod
except:
    mymethod="bfgs"

try:
    mymaxiter=maxiter
except:
    mymaxiter=10000

try:
    myacc=accuracy
except:
    myacc=0.1

try:
    linesearchacc=ls_accuracy
except:
    linesearchacc=1e-5

try:
    # myforcefile="../"+forcefile
    myforcefile=forcefile
    if dip_fit :
        mydipolfile=dipolfile
except:
    print "I need you to set forcefile!"
    sys.exit(1)

alldata=[]
f=open(myforcefile)
if dip_fit :
    d=open(mydipolfile)
try:
    myskip=skipframes
except:
    myskip=1

if fileformat==0:
    for i in xrange(nframes):
        for iskip in xrange(myskip):
            coords=[]
            forces=[]
            data=f.readline().split()
            hmat=[]
            for k in xrange(9):
                hmat.append(float(data[1+k]))
            # Set more reasonable scale of forces
            fs=1e10
            for j in xrange(natoms):
                data=f.readline().split()
                coords.append((float(data[0]),float(data[1]),float(data[2])))
                forces.append((float(data[3])*fs,float(data[4])*fs,float(data[5])*fs))
            if iskip==myskip-1:
                alldata.append((hmat,coords,forces))
elif fileformat==1:
    hmat=newlist(9)
    hmat[0]=boxlen
    hmat[4]=boxlen
    hmat[8]=boxlen
    btoa=0.529177
    # Force unit is Hartree/bohr
    # I want kJ/mol/m
    fs=627.51*4.184*1e3*1e10/6.0221367e23/0.529177
    # Now I want a more reasonable scale of forces
    fs*=1e10
    for i in xrange(nframes):
        for iskip in xrange(myskip):
            coords=[]
            forces=[]
            f.readline() # Dummy
            f.readline() # Dummy
            coordi=[]
            forcei=[]
            for j in xrange(natoms):
                line=f.readline()
                if iskip==myskip-1:
                    data=line.split()
                    coordi.append((float(data[1])*btoa,float(data[2])*btoa,float(data[3])*btoa))
                    forcei.append((float(data[4])*fs,float(data[5])*fs,float(data[6])*fs))
            # Reorder molecules
            if iskip==myskip-1:
                coords.append(coordi[0])
                forces.append(forcei[0])
                nwat=(natoms-1)/3
                for j in xrange(nwat):
                    for k in [1+j,1+nwat+j*2,1+nwat+j*2+1]:
                        coords.append(coordi[k])
                        forces.append(forcei[k])
                alldata.append((hmat,coords,forces))
                
# Per fer la prova amb traject generada per un programa md meu                
elif fileformat==2:
    for i in xrange(nframes):
        for iskip in xrange(myskip):
            coords=[]
            forces=[]
            hmat=[]
            # Set more reasonable scale of forces
            # fs=1e10
            fs = 1e-5
            for j in xrange(natoms):
                data=f.readline().split()
                coords.append((float(data[0]),float(data[1]),float(data[2])))
                forces.append((float(data[3])*fs,float(data[4])*fs,float(data[5])*fs))
            if iskip==myskip-1:
                if dip_fit :
                    alldata.append((hmat,coords,forces,mdips))
                else :
                    alldata.append((hmat,coords,forces))
                
# Traject aigua pura passada directament marco                
elif fileformat==3:
    hmat=[]
    # Force unit is Hartree/bohr    
    btoa=0.529177
    fs=2625.4996251/btoa  # Forces in KJ/(mol A)
    #fs=1   # out forces marco already in au. --> in the same units as cpmd out
    for i in xrange(nframes):
        for iskip in xrange(myskip):
            coords=[]
            forces=[]
            dipols=[]
            coordi=[]
            forcei=[]
            dipoli=[]
            for j in xrange(natoms):
                line=f.readline()
                if iskip==myskip-1:
                    data=line.split()
                    coordi.append((float(data[1])*btoa,float(data[2])*btoa,float(data[3])*btoa))
                    forcei.append((float(data[7])*fs,float(data[8])*fs,float(data[9])*fs))
            if dip_fit:
                for j in xrange(natoms/3):
                    line=d.readline()
                    if iskip==myskip-1:
                        data=line.split()
                        dipoli.append((float(data[1]),float(data[2]),float(data[3])))
                            
            # Reorder molecules
            if iskip==myskip-1:
#                coords.append(coordi[0])
#                forces.append(forcei[0])
                nwat=(natoms)/3
                for j in xrange(nwat):
                    # fixup molecular orientation
                    mc=[]
                    for k in [j,nwat+j*2,nwat+j*2+1]:
                        mc.append(coordi[k])
                        forces.append(forcei[k])
                    if dip_fit:
                        dipols.append(dipoli[j])
                    # Move oxygen into primary cell
                    mcfix=[]
                    mcfix.append((moveatom(mc[0][0],boxlen),moveatom(mc[0][1],boxlen),moveatom(mc[0][2],boxlen)))
                    for k in xrange(2):
                        mcfix.append((minimg(mcfix[0][0],mc[1+k][0],boxlen),minimg(mcfix[0][1],mc[1+k][1],boxlen),minimg(mcfix[0][2],mc[1+k][2],boxlen)))
                    for k in xrange(3):
                        coords.append(mcfix[k])
                    #for k in [j,nwat+j*2,nwat+j*2+1]:
                    #    coords.append(coordi[k])
                    #    forces.append(forcei[k])
                if dip_fit :
                    alldata.append((hmat,coords,forces,dipols))
                else :
                    alldata.append((hmat,coords,forces))

else:
    print "Unknown fileformat for forces and coordinates"
    sys.exit(1)

f.close()

for i in xrange(len(alldata)):
    f=open("restart.res."+`i`,"w")
    haspopcall=0
    try:
        popcall=populate_more_sites
        haspopcall=1
    except:
        alldataX=alldata[i][1]
    if haspopcall:
        alldataX=popcall(alldata[i][1])

#    f.write(`4*len(alldataX)`+" 0\n")
    for j in alldataX:
        for k in xrange(3):
            f.write(`j[k]`+"\n")
#        f.write("0\n")
    f.close()

fevalprivate=[0,alldata,modelfiles,fitchar]

precond=params[:]
for i in xrange(len(params)):
    params[i]=1.

if mymethod=="bfgs":
    bfgs(params,feval,feval_value,mymaxiter,myacc,fevalprivate)
elif mymethod=="polak":
    polak_ribiere(params,feval,feval_value,mymaxiter,myacc,fevalprivate)
elif mymethod=="steepest":
    steepest_descent(params,feval,feval_value,mymaxiter,myacc,fevalprivate)
else:
    print "Unknown optimization method:"+mymethod
    sys.exit(1)

os.chdir("..")
os.mkdir("resultdir")

os.system("cp *.inp resultdir")
os.chdir("resultdir")

replace_parameters(modelfiles,params,fitchar)

os.chdir("..")


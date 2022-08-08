from numpy import append
from sqlalchemy import null
import torch
from torch.autograd import Function

from .util import bger, expandParam, extract_nBatch
from . import solvers
from .solvers.pdipm import batch as pdipm_b
from .solvers.pdipm import spbatch as pdipm_spb
# from .solvers.pdipm import single as pdipm_s

from enum import Enum


class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2


def QPFunction(eps=1e-12, verbose=0, notImprovedLim=3,
                 maxIter=20, solver=QPSolvers.PDIPM_BATCHED,
                 check_Q_spd=True):
    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            """Solve a batch of QPs.

            This function solves a batch of QPs, each optimizing over
            `nz` variables and having `nineq` inequality constraints
            and `neq` equality constraints.
            The optimization problem for each instance in the batch
            (dropping indexing from the notation) is of the form

                \hat z =   argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b

            where Q \in S^{nz,nz},
                S^{nz,nz} is the set of all positive semi-definite matrices,
                p \in R^{nz}
                G \in R^{nineq,nz}
                h \in R^{nineq}
                A \in R^{neq,nz}
                b \in R^{neq}

            These parameters should all be passed to this function as
            Variable- or Parameter-wrapped Tensors.
            (See torch.autograd.Variable and torch.nn.parameter.Parameter)

            If you want to solve a batch of QPs where `nz`, `nineq` and `neq`
            are the same, but some of the contents differ across the
            minibatch, you can pass in tensors in the standard way
            where the first dimension indicates the batch example.
            This can be done with some or all of the coefficients.

            You do not need to add an extra dimension to coefficients
            that will not change across all of the minibatch examples.
            This function is able to infer such cases.

            If you don't want to use any equality or inequality constraints,
            you can set the appropriate values to:

                e = Variable(torch.Tensor())

            Parameters:
            Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
            p:  A (nBatch, nz) or (nz) Tensor.
            G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
            h:  A (nBatch, nineq) or (nineq) Tensor.
            A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
            b:  A (nBatch, neq) or (neq) Tensor.

            Returns: \hat z: a (nBatch, nz) Tensor.
            """
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            #if check_Q_spd:
            #    for i in range(nBatch):
            #        e, _ = torch.eig(Q[i])
            #        if not torch.all(e[:,0] > 0):
            #            raise RuntimeError('Q is not SPD.')

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert(neq > 0 or nineq > 0)
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            if solver == QPSolvers.PDIPM_BATCHED:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
                zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                    Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
                    eps, verbose, notImprovedLim, maxIter)
            elif solver == QPSolvers.CVXPY:
                vals = torch.Tensor(nBatch).type_as(Q)
                zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
                lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) \
                    if ctx.neq > 0 else torch.Tensor()
                slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                for i in range(nBatch):
                    Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                    vals[i], zhati, nui, lami, si = solvers.cvxpy.forward_single_np(
                        *[x.cpu().numpy() if x is not None else None
                        for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
                    # if zhati[0] is None:
                    #     import IPython, sys; IPython.embed(); sys.exit(-1)
                    zhats[i] = torch.Tensor(zhati)
                    lams[i] = torch.Tensor(lami)
                    slacks[i] = torch.Tensor(si)
                    if neq > 0:
                        nus[i] = torch.Tensor(nui)

                ctx.vals = vals
                ctx.lams = lams
                ctx.nus = nus
                ctx.slacks = slacks
            else:
                assert False

            ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            zhats, Q, p, G, h, A, b = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, G, h, A, b)
            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            h, h_e = expandParam(h, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)

            #for i in range(zhats[0].shape[0]):
                #print(i,100.0*(zhats[0][i].item()-h[0][i].item()),ctx.lams[0][i].item())
                #print(i,10.0*(ctx.slacks[0][i].item()),ctx.lams[0][i].item())
                #print(ctx.slacks[0][i].item())

            if solver == QPSolvers.PDIPM_BATCHED:
                zhats_15, nus_15, lams_15, slacks_15 = pdipm_b.forward(
                    Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
                    eps, verbose, notImprovedLim, 15)
            else:
                assert False

            #Compute the constraints which are on the boundary of strict active and inactive
            tor=100
            boundary_list=[]
            for j in range(zhats_15.shape[0]):
                list_possible_bounary=[]
                value=tor
                for i in range(zhats_15[j].shape[0]):
                    a=(1e-10+10.0*(zhats_15[j][i].item()-h[j][i].item()))/(lams_15[j][i].item()+1e-10)
                    b=1/(1e-10+10.0*(zhats_15[j][i].item()-h[j][i].item()))*(lams_15[j][i].item()+1e-10)
                    if abs(a)+abs(b)<tor:
                        list_possible_bounary.append(i)
                print(list_possible_bounary)
                boundary_list.append(list_possible_bounary)
            #print(boundary_list)

            lams1_list=[ctx.lams.clone() for j in range(4) ]
            lams2_list=[ctx.lams.clone() for j in range(4) ]
            slacks1_list=[ctx.lams.clone() for j in range(4) ]
            slacks2_list=[ctx.lams.clone() for j in range(4) ]

            for j in range(zhats.shape[0]):
                if len(boundary_list[j])==1:
                    lams1_list[0].data[j,boundary_list[j][0]]=0
                    slacks2_list[0].data[j,boundary_list[j][0]]=0
                    #lams2_list[0].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[0].data[j,boundary_list[j][0]]=-0.1

                    lams1_list[1].data[j,boundary_list[j][0]]=0
                    slacks2_list[1].data[j,boundary_list[j][0]]=0
                    #lams2_list[1].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[1].data[j,boundary_list[j][0]]=-0.1

                    lams1_list[2].data[j,boundary_list[j][0]]=0
                    slacks2_list[2].data[j,boundary_list[j][0]]=0
                    #lams2_list[2].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[2].data[j,boundary_list[j][0]]=-0.1

                    lams1_list[3].data[j,boundary_list[j][0]]=0
                    slacks2_list[3].data[j,boundary_list[j][0]]=0
                    #lams2_list[3].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[3].data[j,boundary_list[j][0]]=-0.1
                elif len(boundary_list[j])==2:
                    lams1_list[0].data[j,boundary_list[j][0]]=0
                    slacks2_list[0].data[j,boundary_list[j][0]]=0
                    #lams2_list[0].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[0].data[j,boundary_list[j][0]]=-0.1

                    lams1_list[1].data[j,boundary_list[j][1]]=0
                    slacks2_list[1].data[j,boundary_list[j][1]]=0
                    #lams2_list[1].data[j,boundary_list[j][1]]=0.1
                    #slacks1_list[1].data[j,boundary_list[j][1]]=-0.1

                    lams1_list[2].data[j,boundary_list[j][0]]=0
                    slacks2_list[2].data[j,boundary_list[j][0]]=0
                    #lams2_list[2].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[2].data[j,boundary_list[j][0]]=-0.1

                    lams1_list[3].data[j,boundary_list[j][1]]=0
                    slacks2_list[3].data[j,boundary_list[j][1]]=0
                    #lams2_list[3].data[j,boundary_list[j][1]]=0.1
                    #slacks1_list[3].data[j,boundary_list[j][1]]=-0.1

                elif len(boundary_list[j])==3:
                    lams1_list[0].data[j,boundary_list[j][0]]=0
                    slacks2_list[0].data[j,boundary_list[j][0]]=0
                    #lams2_list[0].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[0].data[j,boundary_list[j][0]]=-0.1

                    lams1_list[1].data[j,boundary_list[j][1]]=0
                    slacks2_list[1].data[j,boundary_list[j][1]]=0
                    #lams2_list[1].data[j,boundary_list[j][1]]=0.1
                    #slacks1_list[1].data[j,boundary_list[j][1]]=-0.1

                    lams1_list[2].data[j,boundary_list[j][2]]=0
                    slacks2_list[2].data[j,boundary_list[j][2]]=0
                    #lams2_list[2].data[j,boundary_list[j][2]]=0.1
                    #slacks1_list[2].data[j,boundary_list[j][2]]=-0.1

                elif len(boundary_list[j])>=4:
                    lams1_list[0].data[j,boundary_list[j][0]]=0
                    slacks2_list[0].data[j,boundary_list[j][0]]=0
                    #lams2_list[0].data[j,boundary_list[j][0]]=0.1
                    #slacks1_list[0].data[j,boundary_list[j][0]]=-0.1

                    lams1_list[1].data[j,boundary_list[j][1]]=0
                    slacks2_list[1].data[j,boundary_list[j][1]]=0
                    #lams2_list[1].data[j,boundary_list[j][1]]=0.1
                    #slacks1_list[1].data[j,boundary_list[j][1]]=-0.1

                    lams1_list[2].data[j,boundary_list[j][2]]=0
                    slacks2_list[2].data[j,boundary_list[j][2]]=0
                    #lams2_list[2].data[j,boundary_list[j][2]]=0.1
                    #slacks1_list[2].data[j,boundary_list[j][2]]=-0.1

                    lams1_list[3].data[j,boundary_list[j][3]]=0
                    slacks2_list[3].data[j,boundary_list[j][3]]=0
                    #lams2_list[3].data[j,boundary_list[j][3]]=0.1
                    #slacks1_list[3].data[j,boundary_list[j][3]]=-0.1

            # neq, nineq, nz = ctx.neq, ctx.nineq, ctx.nz
            neq, nineq = ctx.neq, ctx.nineq

            if solver == QPSolvers.CVXPY:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)

            '''
            # Clamp here to avoid issues coming up when the slacks are too small.
            # TODO: A better fix would be to get lams and slacks from the
            # solver that don't have this issue.
            d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)
            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)
            dx, _, dlam, dnu = pdipm_b.solve_kkt(
                ctx.Q_LU, d, G, A, ctx.S_LU,
                dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())
            dps = dx
            dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
            if G_e:
                dGs = dGs.mean(0)
            dhs = -dlam
            if h_e:
                dhs = dhs.mean(0)
            if neq > 0:
                dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                dbs = -dnu
                if A_e:
                    dAs = dAs.mean(0)
                if b_e:
                    dbs = dbs.mean(0)
            else:
                dAs, dbs = None, None
            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            if Q_e:
                dQs = dQs.mean(0)
            if p_e:
                dps = dps.mean(0)
            grads = (dQs, dps, dGs, dhs, dAs, dbs)
            '''
            dQs_list=[]
            for i in range(4):
                d1 = torch.clamp(lams1_list[i], min=1e-8) / torch.clamp(slacks1_list[i], min=1e-8)
                pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d1)
                dx1, _, dlam1, dnu1 = pdipm_b.solve_kkt(
                    ctx.Q_LU, d1, G, A, ctx.S_LU,
                    dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
                    torch.zeros(nBatch, nineq).type_as(G),
                    torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())
                dps1 = dx1
                dGs1 = bger(dlam1, zhats) + bger(lams1_list[i], dx1)
                if G_e:
                    dGs1 = dGs1.mean(0)
                dhs1 = -dlam1
                if h_e:
                    dhs1 = dhs1.mean(0)
                if neq > 0:
                    dAs1 = bger(dnu1, zhats) + bger(ctx.nus, dx1)
                    dbs1 = -dnu1
                    if A_e:
                        dAs1 = dAs1.mean(0)
                    if b_e:
                        dbs1 = dbs1.mean(0)
                else:
                    dAs1, dbs1 = None, None
                dQs1 = 0.5 * (bger(dx1, zhats) + bger(zhats, dx1))
                if Q_e:
                    dQs1 = dQs1.mean(0)
                if p_e:
                    dps1 = dps1.mean(0)
                grads1 = (dQs1, dps1, dGs1, dhs1, dAs1, dbs1)

                d2 = torch.clamp(lams2_list[i], min=1e-8) / torch.clamp(slacks2_list[i], min=1e-8)
                pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d2)
                dx2, _, dlam2, dnu2 = pdipm_b.solve_kkt(
                    ctx.Q_LU, d2, G, A, ctx.S_LU,
                    dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
                    torch.zeros(nBatch, nineq).type_as(G),
                    torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())
                dps2 = dx2
                dGs2 = bger(dlam2, zhats) + bger(lams2_list[i], dx2)
                if G_e:
                    dGs2 = dGs2.mean(0)
                dhs2 = -dlam2
                if h_e:
                    dhs2 = dhs2.mean(0)
                if neq > 0:
                    dAs2 = bger(dnu2, zhats) + bger(ctx.nus, dx2)
                    dbs2 = -dnu2
                    if A_e:
                        dAs2 = dAs2.mean(0)
                    if b_e:
                        dbs2 = dbs2.mean(0)
                else:
                    dAs2, dbs2 = None, None
                dQs2 = 0.5 * (bger(dx2, zhats) + bger(zhats, dx2))
                if Q_e:
                    dQs2 = dQs2.mean(0)
                if p_e:
                    dps2 = dps2.mean(0)
                grads2 = (dQs2, dps2, dGs2, dhs2, dAs2, dbs2)
                dQs_list.append(dQs1.clone())
                dQs_list.append(dQs2.clone())

            grads=(sum(dQs_list)*0.125, 0.5*dps1+0.5*dps2, 0.5*dGs1+0.5*dGs2, 0.5*dhs1+0.5*dhs2, 0.5*dAs1+0.5*dAs2, 0.5*dbs1+0.5*dbs2)
            return grads
    return QPFunctionFn.apply

class SpQPFunction(Function):
    def __init__(self, Qi, Qsz, Gi, Gsz, Ai, Asz,
                 eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
        self.Qi, self.Qsz = Qi, Qsz
        self.Gi, self.Gsz = Gi, Gsz
        self.Ai, self.Asz = Ai, Asz

        self.eps = eps
        self.verbose = verbose
        self.notImprovedLim = notImprovedLim
        self.maxIter = maxIter

        self.nineq, self.nz = Gsz
        self.neq, _ = Asz

    def forward(self, Qv, p, Gv, h, Av, b):
        self.nBatch = Qv.size(0)

        zhats, self.nus, self.lams, self.slacks = pdipm_spb.forward(
            self.Qi, Qv, self.Qsz, p, self.Gi, Gv, self.Gsz, h,
            self.Ai, Av, self.Asz, b, self.eps, self.verbose,
            self.notImprovedLim, self.maxIter)

        self.save_for_backward(zhats, Qv, p, Gv, h, Av, b)
        return zhats

    def backward(self, dl_dzhat):
        zhats, Qv, p, Gv, h, Av, b = self.saved_tensors

        Di = type(self.Qi)([range(self.nineq), range(self.nineq)])
        Dv = self.lams / self.slacks
        Dsz = torch.Size([self.nineq, self.nineq])
        dx, _, dlam, dnu = pdipm_spb.solve_kkt(
            self.Qi, Qv, self.Qsz, Di, Dv, Dsz,
            self.Gi, Gv, self.Gsz,
            self.Ai, Av, self.Asz, dl_dzhat,
            type(p)(self.nBatch, self.nineq).zero_(),
            type(p)(self.nBatch, self.nineq).zero_(),
            type(p)(self.nBatch, self.neq).zero_())

        dps = dx

        dGs = bger(dlam, zhats) + bger(self.lams, dx)
        GM = torch.cuda.sparse.DoubleTensor(
            self.Gi, Gv[0].clone().fill_(1.0), self.Gsz
        ).to_dense().byte().expand_as(dGs)
        dGs = dGs[GM].view_as(Gv)

        dhs = -dlam

        dAs = bger(dnu, zhats) + bger(self.nus, dx)
        AM = torch.cuda.sparse.DoubleTensor(
            self.Ai, Av[0].clone().fill_(1.0), self.Asz
        ).to_dense().byte().expand_as(dAs)
        dAs = dAs[AM].view_as(Av)

        dbs = -dnu

        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
        QM = torch.cuda.sparse.DoubleTensor(
            self.Qi, Qv[0].clone().fill_(1.0), self.Qsz
        ).to_dense().byte().expand_as(dQs)
        dQs = dQs[QM].view_as(Qv)

        grads = (dQs, dps, dGs, dhs, dAs, dbs)

        return grads

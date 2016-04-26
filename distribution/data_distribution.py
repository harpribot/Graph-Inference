import pandas as pd
import numpy as np
from math import log10

def get_distribution(df):

    D_vals = np.arange(3)
    I_vals = np.arange(3)
    G_vals = np.arange(3)
    R_vals = np.arange(3)
    S_vals = np.arange(3)

    # P(D)
    P_D = np.zeros([D_vals.size,1])

    for d in D_vals:
        P_D[d] = df.loc[df['D'] == d].shape[0]/float(df.shape[0])

    # P(I)
    P_I = np.zeros([I_vals.size,1])

    for i in I_vals:
        P_I[i] = df.loc[df['I'] == i].shape[0]/float(df.shape[0])

    # P(G | D, I)
    P_DI_G = np.zeros([D_vals.size, I_vals.size,G_vals.size])

    for d in D_vals:
        df_d = df.loc[df['D'] == d]
        for i in I_vals:
            df_di = df_d.loc[df_d['I'] == i]
            for g in G_vals:
                df_di_g = df_di.loc[df_di['G'] == g]
                if df_di.shape[0] != 0:
                    P_DI_G[d,i,g] = df_di_g.shape[0]/float(df_di.shape[0])
                else:
                    P_DI_G[d,i,g] = 0


    # P(S | I)
    P_I_S = np.zeros([I_vals.size,S_vals.size])

    for i in I_vals:
        df_i = df.loc[df['I'] == i]
        for s in S_vals:
            df_s_i = df_i.loc[df_i['S'] == s]
            if df_i.shape[0] != 0:
                P_I_S[i,s] = df_s_i.shape[0]/float(df_i.shape[0])
            else:
                P_I_S[i,s] = 0

    # P(R | G)
    P_G_R = np.zeros([G_vals.size,R_vals.size])

    for g in G_vals:
        df_g = df.loc[df['G'] == g]
        for r in R_vals:
            df_r_g = df_g.loc[df_g['R'] == r]
            if df_g.shape[0] != 0:
                P_G_R[g,r] = df_r_g.shape[0]/float(df_g.shape[0])
            else:
                P_G_R[g,r] = 0


    return P_I, P_D,P_DI_G, P_G_R, P_I_S

def reconstruct_dist(map_DI_G, map_DI, map_R_G, map_G):
    P_DI_G = np.zeros([3,3,3])
    P_G_R = np.zeros([3,3])

    # Reconstruct P_G_DI
    for d in range(3):
        for i in range(3):
            for g in range(3):
                if map_DI[(d,i)] != 0:
                    P_DI_G[d,i,g] = map_DI_G[(d,i)][g]/float(map_DI[(d,i)])
                else:
                    P_DI_G[d,i,g] = 0.

    # Reconstruct P_R_G
    for r in range(3):
        for g in range(3):
            if map_G[1][g] != 0:
                P_G_R[g,r] = map_R_G[r][g]/float(map_G[1][g])
            else:
                P_G_R[g,r] = 0.


    return P_DI_G, P_G_R

def get_log_likelihood(d,i,g_new,s,r, P_D, P_I, P_DI_G, P_I_S, P_G_R):
    P_d = P_D[d]
    P_i = P_I[i]
    P_di_g = P_DI_G[d,i,g_new]
    P_i_s = P_I_S[i,s]
    P_g_r = P_G_R[g_new,r]

    return log10(P_d) + log10(P_i) + log10(P_di_g) + log10(P_i_s) + log10(P_g_r)

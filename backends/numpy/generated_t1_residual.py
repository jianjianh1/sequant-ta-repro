import numpy as np
import os

def whole_t1_residual():

    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    f_ii = np.load('f_ii.npy')
    I2_iÎ += np.einsum('iÎ,ia->aÎ', I3_iÎ, f_ii, optimize=True)
    del f_ii
    del I3_iÎ
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -1 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I2_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I2_iÎ
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    I_iÎ += np.einsum('Î,aiÎ->ia', I_Î, g_ÎiÎ, optimize=True)
    del g_ÎiÎ
    del I_Î
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += 2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ÎÎ = np.zeros((dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I2_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I2_iÎ
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_ÎÎ += np.einsum('Î,abÎ->ab', I_Î, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_Î
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_iÎ += np.einsum('Îa,ia->iÎ', I_ÎÎ, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += 2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I2_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I2_iÎ
    g_iiÎ = np.load('g_iiÎ.npy')
    I_ii += np.einsum('Î,iaÎ->ai', I_Î, g_iiÎ, optimize=True)
    del g_iiÎ
    del I_Î
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I2_iÎ += np.einsum('iÎ,aÎ->ia', I3_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I3_iÎ
    I_iÎ += np.einsum('ia,aÎ->iÎ', I_ii, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ii
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I3_ia = np.zeros((dim_i, dim_a), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I4_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I4_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I4_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I4_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I3_iÎ += np.einsum('Î,iaÎ->ia', I_Î, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_Î
    C_Îa = np.load('C_Îa.npy')
    I3_ia += np.einsum('iÎ,Îa->ia', I3_iÎ, C_Îa, optimize=True)
    del C_Îa
    del I3_iÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('ia,abic->cb', I3_ia, t_aaii, optimize=True)
    del t_aaii
    del I3_ia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += 4 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I3_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I3_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I2_iÎ += np.einsum('Î,iaÎ->ia', I_Î, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_Î
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_ii += np.einsum('iÎ,aÎ->ai', I2_iÎ, I3_iÎ, optimize=True)
    del I3_iÎ
    del I2_iÎ
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I2_iÎ += np.einsum('iÎ,aÎ->ia', I3_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I3_iÎ
    I_iÎ += np.einsum('ia,aÎ->iÎ', I_ii, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ii
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I3_ia = np.zeros((dim_i, dim_a), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I4_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I4_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I4_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I4_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I3_iÎ += np.einsum('Î,iaÎ->ia', I_Î, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_Î
    C_Îa = np.load('C_Îa.npy')
    I3_ia += np.einsum('iÎ,Îa->ia', I3_iÎ, C_Îa, optimize=True)
    del C_Îa
    del I3_iÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('ia,abci->cb', I3_ia, t_aaii, optimize=True)
    del t_aaii
    del I3_ia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I2_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I2_iÎ
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎÎ += np.einsum('iÎ,aÎb->iab', I2_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I2_iÎ
    I_iÎ += np.einsum('iaÎ,abÎ->ib', I_iiÎ, I_iÎÎ, optimize=True)
    del I_iÎÎ
    del I_iiÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -1 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I3_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I3_iÎ
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I_iiia += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, I_iaÎ, optimize=True)
    del I_iaÎ
    del I_iiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('iabc,cdba->id', I_iiia, t_aaii, optimize=True)
    del t_aaii
    del I_iiia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I_iiaÎ += np.einsum('Îa,baic->cibÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I_iaÎ, I_iiaÎ, optimize=True)
    del I_iiaÎ
    del I_iaÎ
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎ += np.einsum('iÎa,bÎa->ib', I_iÎÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += 2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I_iiaÎ += np.einsum('Îa,baic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I_iaÎ, I_iiaÎ, optimize=True)
    del I_iiaÎ
    del I_iaÎ
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎ += np.einsum('iÎa,bÎa->ib', I_iÎÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -1 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I3_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I3_iÎ
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I_iiia += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, I_iaÎ, optimize=True)
    del I_iaÎ
    del I_iiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('iabc,cdab->id', I_iiia, t_aaii, optimize=True)
    del t_aaii
    del I_iiia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I3_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I3_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I2_iÎ += np.einsum('iaÎ,ibÎ->ab', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_ii += np.einsum('iÎ,aÎ->ai', I2_iÎ, I3_iÎ, optimize=True)
    del I3_iÎ
    del I2_iÎ
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I2_iÎ += np.einsum('iÎ,aÎ->ia', I3_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I3_iÎ
    I_iÎ += np.einsum('ia,aÎ->iÎ', I_ii, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ii
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I3_ia = np.zeros((dim_i, dim_a), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I4_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I4_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->aib', I4_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I4_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I3_iÎ += np.einsum('iaÎ,abÎ->ib', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_ia += np.einsum('iÎ,Îa->ia', I3_iÎ, C_Îa, optimize=True)
    del C_Îa
    del I3_iÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('ia,abic->cb', I3_ia, t_aaii, optimize=True)
    del t_aaii
    del I3_ia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I3_ia = np.zeros((dim_i, dim_a), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I4_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I4_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->aib', I4_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I4_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I3_iÎ += np.einsum('iaÎ,abÎ->ib', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_ia += np.einsum('iÎ,Îa->ia', I3_iÎ, C_Îa, optimize=True)
    del C_Îa
    del I3_iÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('ia,abci->cb', I3_ia, t_aaii, optimize=True)
    del t_aaii
    del I3_ia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎÎ += np.einsum('iÎa,bca->ibÎc', g_iÎÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_iiaÎ += np.einsum('iaÎb,Îc->iacb', I_iiÎÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    t_aaii = np.load('t_aaii.npy')
    I_ii += np.einsum('iabc,bcdi->da', I_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I_iiaa
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I2_iÎ += np.einsum('iÎ,aÎ->ia', I3_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I3_iÎ
    I_iÎ += np.einsum('ia,aÎ->iÎ', I_ii, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ii
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎÎ += np.einsum('iÎa,bca->ibÎc', g_iÎÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_iiaÎ += np.einsum('iaÎb,Îc->iacb', I_iiÎÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    t_aaii = np.load('t_aaii.npy')
    I_ii += np.einsum('iabc,bcda->di', I_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I_iiaa
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I2_iÎ += np.einsum('iÎ,aÎ->ia', I3_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I3_iÎ
    I_iÎ += np.einsum('ia,aÎ->iÎ', I_ii, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ii
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_ii += np.einsum('iabÎ,aÎ->ib', I_iiiÎ, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_iiiÎ
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I2_iÎ += np.einsum('iÎ,aÎ->ia', I3_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I3_iÎ
    I_iÎ += np.einsum('ia,aÎ->iÎ', I_ii, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ii
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    f_iÎ = np.load('f_iÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_ia += np.einsum('iÎ,Îa->ia', f_iÎ, C_Îa, optimize=True)
    del C_Îa
    del f_iÎ
    t_ai = np.load('t_ai.npy')
    I_ii += np.einsum('ia,ab->bi', I2_ia, t_ai, optimize=True)
    del t_ai
    del I2_ia
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I3_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I3_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I2_iÎ += np.einsum('iÎ,aÎ->ia', I3_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I3_iÎ
    I_iÎ += np.einsum('ia,aÎ->iÎ', I_ii, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_ii
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -1 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iiÎÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del g_iiÎ
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_iÎ += np.einsum('iaÎb,ab->iÎ', I_iiÎÎ, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_iiÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -1 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    f_ÎÎ = np.load('f_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, f_ÎÎ, optimize=True)
    del f_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('iabc,cdba->id', I_iiia, t_aaii, optimize=True)
    del t_aaii
    del I_iiia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('iabc,cdab->id', I_iiia, t_aaii, optimize=True)
    del t_aaii
    del I_iiia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I3_ia = np.zeros((dim_i, dim_a), order='F')
    f_iÎ = np.load('f_iÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I3_ia += np.einsum('iÎ,Îa->ia', f_iÎ, C_Îa, optimize=True)
    del C_Îa
    del f_iÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('ia,abci->cb', I3_ia, t_aaii, optimize=True)
    del t_aaii
    del I3_ia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += -1 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    I3_ia = np.zeros((dim_i, dim_a), order='F')
    f_iÎ = np.load('f_iÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I3_ia += np.einsum('iÎ,Îa->ia', f_iÎ, C_Îa, optimize=True)
    del C_Îa
    del f_iÎ
    t_aaii = np.load('t_aaii.npy')
    I2_ia += np.einsum('ia,abic->cb', I3_ia, t_aaii, optimize=True)
    del t_aaii
    del I3_ia
    C_Îa = np.load('C_Îa.npy')
    I2_iÎ += np.einsum('ia,Îa->iÎ', I2_ia, C_Îa, optimize=True)
    del C_Îa
    del I2_ia
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += 2 * np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    f_Îi = np.load('f_Îi.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('Îi,aÎ->ia', f_Îi, C_aÎ, optimize=True)
    del C_aÎ
    del f_Îi
    np.save('I_ia.npy', I_ia)


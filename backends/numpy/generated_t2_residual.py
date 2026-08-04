import numpy as np
import os

def whole_t2_residual():

    I_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
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
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iÎ += np.einsum('Î,iaÎ->ia', I_Î, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_Î
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_ii += np.einsum('iÎ,aÎ->ai', I_iÎ, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_iÎ
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I2_iiaa += np.einsum('ia,bc->iabc', I_ii, I_aa, optimize=True)
    del I_aa
    del I_ii
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += -4 * np.einsum('iabc,daec->dieb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    I_ÎÎ = np.zeros((dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_ÎÎ += np.einsum('Î,abÎ->ab', I_Î, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_Î
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', I_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiaa += 4 * np.einsum('ab,cbid->idca', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_ia = np.zeros((dim_i, dim_a), order='F')
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
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iÎ += np.einsum('Î,iaÎ->ia', I_Î, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_Î
    C_Îa = np.load('C_Îa.npy')
    I_ia += np.einsum('iÎ,Îa->ia', I_iÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiia += np.einsum('ia,bacd->cdib', I_ia, t_aaii, optimize=True)
    del t_aaii
    del I_ia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += -4 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_Î = np.zeros((dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_Î += np.einsum('iÎ,iÎa->a', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_iiÎ = np.load('g_iiÎ.npy')
    I_ii += np.einsum('Î,iaÎ->ai', I_Î, g_iiÎ, optimize=True)
    del g_iiÎ
    del I_Î
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I2_iiaa += np.einsum('ia,bc->iabc', I_ii, I_aa, optimize=True)
    del I_aa
    del I_ii
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->diac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += -4 * np.einsum('iabc,daec->ideb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_ia = np.zeros((dim_i, dim_a), order='F')
    f_iÎ = np.load('f_iÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_ia += np.einsum('iÎ,Îa->ia', f_iÎ, C_Îa, optimize=True)
    del C_Îa
    del f_iÎ
    t_ai = np.load('t_ai.npy')
    I_ii += np.einsum('ia,ab->bi', I_ia, t_ai, optimize=True)
    del t_ai
    del I_ia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I2_iiaa += np.einsum('ia,bc->iabc', I_ii, I_aa, optimize=True)
    del I_aa
    del I_ii
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->diac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += -2 * np.einsum('iabc,daec->ideb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
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
    I3_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    I4_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I4_iiaa += np.einsum('ab,cbid->diac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I2_iiaa += np.einsum('iabc,daeb->diec', I3_iiaa, I4_iiaa, optimize=True)
    del I4_iiaa
    del I3_iiaa
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,cbid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += 2 * np.einsum('iabc,daec->idbe', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
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
    I3_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    I4_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I4_iiaa += np.einsum('ab,cbid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I2_iiaa += np.einsum('iabc,daeb->diec', I3_iiaa, I4_iiaa, optimize=True)
    del I4_iiaa
    del I3_iiaa
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,cbid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += np.einsum('iabc,daec->dibe', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I3_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I3_iiaÎ += np.einsum('Îa,abic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I2_iiaÎ += np.einsum('ab,icbÎ->icaÎ', I_aa, I3_iiaÎ, optimize=True)
    del I3_iiaÎ
    del I_aa
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iiÎÎ += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iiÎ
    I_iiaÎ += np.einsum('iabÎ,cadÎ->cibd', I2_iiaÎ, I_iiÎÎ, optimize=True)
    del I_iiÎÎ
    del I2_iiaÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiaa += -2 * np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiaÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    I2_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I2_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    I_iiii += np.einsum('iaÎ,bcÎ->ibac', I_iiÎ, I2_iiÎ, optimize=True)
    del I2_iiÎ
    del I_iiÎ
    I2_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,abic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    C_Îa = np.load('C_Îa.npy')
    I2_iiÎÎ += np.einsum('iabÎ,cb->iaÎc', I2_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I2_iiaÎ
    I_iiÎÎ += np.einsum('iabc,bcÎd->iaÎd', I_iiii, I2_iiÎÎ, optimize=True)
    del I2_iiÎÎ
    del I_iiii
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaÎ += np.einsum('iaÎb,cÎ->iacb', I_iiÎÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiÎÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,abic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    C_Îa = np.load('C_Îa.npy')
    I2_iiÎÎ += np.einsum('iabÎ,cb->iaÎc', I2_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I2_iiaÎ
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    I_iiii += np.einsum('iaÎ,bcÎ->acib', g_iiÎ, g_iiÎ, optimize=True)
    del g_iiÎ
    I_iiÎÎ += np.einsum('iaÎb,cdia->cdÎb', I2_iiÎÎ, I_iiii, optimize=True)
    del I_iiii
    del I2_iiÎÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaÎ += np.einsum('iaÎb,cÎ->iacb', I_iiÎÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiÎÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    I_iiii += np.einsum('iaÎ,bcÎ->acib', g_iiÎ, g_iiÎ, optimize=True)
    del g_iiÎ
    I_iiia += np.einsum('ia,bcid->bcda', I_ia, I_iiii, optimize=True)
    del I_iiii
    del I_ia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    I2_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,abic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    C_Îa = np.load('C_Îa.npy')
    I2_iiÎÎ += np.einsum('iabÎ,cb->iaÎc', I2_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I2_iiaÎ
    I3_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I3_iiÎÎ += np.einsum('iÎa,bca->ibÎc', g_iÎÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    I_iiii += np.einsum('iaÎb,cdÎb->iacd', I2_iiÎÎ, I3_iiÎÎ, optimize=True)
    del I3_iiÎÎ
    del I2_iiÎÎ
    I2_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,abic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    C_Îa = np.load('C_Îa.npy')
    I2_iiÎÎ += np.einsum('iabÎ,cb->iaÎc', I2_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I2_iiaÎ
    I_iiÎÎ += np.einsum('iabc,bcÎd->iaÎd', I_iiii, I2_iiÎÎ, optimize=True)
    del I2_iiÎÎ
    del I_iiii
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaÎ += np.einsum('iaÎb,cÎ->iacb', I_iiÎÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiÎÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I_iiaÎ += np.einsum('Îa,abic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    C_Îa = np.load('C_Îa.npy')
    I_iiÎÎ += np.einsum('iabÎ,cb->iaÎc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    I2_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I2_iiÎÎ += np.einsum('iÎa,bca->ibÎc', g_iÎÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    I_iiii += np.einsum('iaÎb,cdÎb->iacd', I_iiÎÎ, I2_iiÎÎ, optimize=True)
    del I2_iiÎÎ
    del I_iiÎÎ
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiia += np.einsum('iabc,bd->iacd', I_iiii, I_ia, optimize=True)
    del I_ia
    del I_iiii
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I_iiaÎ += np.einsum('Îa,abic->cibÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    C_Îa = np.load('C_Îa.npy')
    I_iiÎÎ += np.einsum('iabÎ,cb->iaÎc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    I_iÎÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎÎÎ += np.einsum('iÎa,bca->iÎbc', g_iÎÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del g_iÎÎ
    I_iiiÎ += np.einsum('iaÎb,cÎdb->iacd', I_iiÎÎ, I_iÎÎÎ, optimize=True)
    del I_iÎÎÎ
    del I_iiÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiia += np.einsum('iabÎ,cÎ->iabc', I_iiiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiiÎ
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += -2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_iiii += np.einsum('iabÎ,cÎ->icab', I_iiiÎ, I_iÎ, optimize=True)
    del I_iÎ
    del I_iiiÎ
    I2_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,abic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    C_Îa = np.load('C_Îa.npy')
    I2_iiÎÎ += np.einsum('iabÎ,cb->iaÎc', I2_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I2_iiaÎ
    I_iiÎÎ += np.einsum('iabc,bcÎd->iaÎd', I_iiii, I2_iiÎÎ, optimize=True)
    del I2_iiÎÎ
    del I_iiii
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaÎ += np.einsum('iaÎb,cÎ->iacb', I_iiÎÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiÎÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += 2 * np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    I2_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I2_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    I_iiii += np.einsum('iaÎ,bcÎ->ibac', I_iiÎ, I2_iiÎ, optimize=True)
    del I2_iiÎ
    del I_iiÎ
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiia += np.einsum('iabc,bd->iacd', I_iiii, I_ia, optimize=True)
    del I_ia
    del I_iiii
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->cibÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I_iaÎ
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    I2_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I2_iÎÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iaÎ += np.einsum('iÎa,bÎ->iba', I2_iÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I2_iÎÎ
    I_iiaÎ += np.einsum('iÎa,bca->ibcÎ', I_iÎÎ, I_iaÎ, optimize=True)
    del I_iaÎ
    del I_iÎÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += 4 * np.einsum('iabÎ,cÎ->iacb', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->cibÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I2_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I2_iaÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iaÎ += np.einsum('iÎa,bÎ->iba', I_iÎÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iÎÎ
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->cibÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I2_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I2_iaÎ
    I_iiaÎ += np.einsum('iaÎ,bcÎ->ibac', I_iaÎ, I_iÎÎ, optimize=True)
    del I_iÎÎ
    del I_iaÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += 4 * np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->cibÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I2_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I2_iaÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iaÎ += np.einsum('iÎa,bÎ->iba', I_iÎÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iÎÎ
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I2_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I2_iaÎ
    I_iiaÎ += np.einsum('iaÎ,bcÎ->ibac', I_iaÎ, I_iÎÎ, optimize=True)
    del I_iÎÎ
    del I_iaÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += -4 * np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I2_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I2_iaÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iaÎ += np.einsum('iÎa,bÎ->iba', I_iÎÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iÎÎ
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I2_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I2_iaÎ
    I_iiaÎ += np.einsum('iaÎ,bcÎ->ibac', I_iaÎ, I_iÎÎ, optimize=True)
    del I_iÎÎ
    del I_iaÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I2_iiaÎ += np.einsum('Îa,baic->icbÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I_iÎÎ += np.einsum('iaÎ,biac->bcÎ', I_iaÎ, I2_iiaÎ, optimize=True)
    del I2_iiaÎ
    del I_iaÎ
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    I2_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I2_iÎÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iaÎ += np.einsum('iÎa,bÎ->iba', I2_iÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I2_iÎÎ
    I_iiaÎ += np.einsum('iÎa,bca->ibcÎ', I_iÎÎ, I_iaÎ, optimize=True)
    del I_iaÎ
    del I_iÎÎ
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    I_iiaa += -2 * np.einsum('iabÎ,cÎ->iacb', I_iiaÎ, I_aÎ, optimize=True)
    del I_aÎ
    del I_iiaÎ
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    I_ÎÎ = np.zeros((dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_ÎÎ = np.zeros((dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiaÎ += np.einsum('iaÎ,bcÎ->biac', I_iaÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iaÎ
    t_aaii = np.load('t_aaii.npy')
    I2_aÎ += np.einsum('iabÎ,bcia->cÎ', I_iiaÎ, t_aaii, optimize=True)
    del t_aaii
    del I_iiaÎ
    C_Îa = np.load('C_Îa.npy')
    I2_ÎÎ += np.einsum('aÎ,ba->Îb', I2_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I2_aÎ
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_ÎÎ += np.einsum('Îa,ba->Îb', I2_ÎÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aÎ += np.einsum('Îa,Îb->ba', I_ÎÎ, C_Îa, optimize=True)
    del C_Îa
    del I_ÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_aa += np.einsum('aÎ,bÎ->ba', I_aÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiaa += 2 * np.einsum('ab,cbid->idca', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    I_ÎÎ = np.zeros((dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_ÎÎ = np.zeros((dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I2_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiaÎ += np.einsum('iaÎ,bcÎ->ibac', I_iaÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iaÎ
    t_aaii = np.load('t_aaii.npy')
    I2_aÎ += np.einsum('iabÎ,bcia->cÎ', I_iiaÎ, t_aaii, optimize=True)
    del t_aaii
    del I_iiaÎ
    C_Îa = np.load('C_Îa.npy')
    I2_ÎÎ += np.einsum('aÎ,ba->Îb', I2_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I2_aÎ
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_ÎÎ += np.einsum('Îa,ba->Îb', I2_ÎÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aÎ += np.einsum('Îa,Îb->ba', I_ÎÎ, C_Îa, optimize=True)
    del C_Îa
    del I_ÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_aa += np.einsum('aÎ,bÎ->ba', I_aÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiaa += -4 * np.einsum('ab,cbid->idca', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I2_iaÎ += np.einsum('Îia,bÎ->iba', g_ÎiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del g_ÎiÎ
    I3_iiaa += np.einsum('iaÎ,bcÎ->bica', I_iaÎ, I2_iaÎ, optimize=True)
    del I2_iaÎ
    del I_iaÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiaa += np.einsum('iabc,cdea->iebd', I3_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I3_iiaa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiaa += -2 * np.einsum('iabc,dc->iabd', I2_iiaa, I_aa, optimize=True)
    del I_aa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I_iaÎ += np.einsum('iÎa,Îb->iba', g_iÎÎ, C_Îa, optimize=True)
    del C_Îa
    del g_iÎÎ
    I2_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I2_iaÎ += np.einsum('Îia,bÎ->iba', g_ÎiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del g_ÎiÎ
    I3_iiaa += np.einsum('iaÎ,bcÎ->bica', I_iaÎ, I2_iaÎ, optimize=True)
    del I2_iaÎ
    del I_iaÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiaa += np.einsum('iabc,cdae->iebd', I3_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I3_iiaa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiaa += 4 * np.einsum('iabc,dc->iabd', I2_iiaa, I_aa, optimize=True)
    del I_aa
    del I2_iiaa
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I2_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I3_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_aaii = np.load('t_aaii.npy')
    I3_iiaÎ += np.einsum('Îa,abic->cibÎ', C_Îa, t_aaii, optimize=True)
    del t_aaii
    del C_Îa
    I2_iiaÎ += np.einsum('ab,icbÎ->icaÎ', I_aa, I3_iiaÎ, optimize=True)
    del I3_iiaÎ
    del I_aa
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iiÎÎ += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iiÎ
    I_iiaÎ += np.einsum('iabÎ,cadÎ->cibd', I2_iiaÎ, I_iiÎÎ, optimize=True)
    del I_iiÎÎ
    del I2_iiaÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiaa += -2 * np.einsum('iabÎ,cÎ->iacb', I_iiaÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiaÎ
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdeb->eiad', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdbe->eiad', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += -4 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdea->eibd', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iadc', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->iabc', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdae->eibd', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_iaÎ += np.einsum('Îia,bÎ->iba', g_ÎiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del g_ÎiÎ
    I_iiia += np.einsum('iaÎ,bcÎ->ibac', I_iiÎ, I_iaÎ, optimize=True)
    del I_iaÎ
    del I_iiÎ
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += -2 * np.einsum('iabc,bd->iadc', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎ
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_iaÎ += np.einsum('Îia,bÎ->iba', g_ÎiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del g_ÎiÎ
    I_iiaÎ += np.einsum('iÎa,bca->bicÎ', I_iÎÎ, I_iaÎ, optimize=True)
    del I_iaÎ
    del I_iÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiaa += 2 * np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiaÎ
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_iaÎ += np.einsum('Îia,bÎ->iba', g_ÎiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del g_ÎiÎ
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    I_iiaÎ += np.einsum('iaÎ,bcÎ->icab', I_iaÎ, g_ÎiÎ, optimize=True)
    del g_ÎiÎ
    del I_iaÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiaa += np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiaÎ
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->aib', I2_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I2_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iÎ += np.einsum('iaÎ,abÎ->ib', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I_ia += np.einsum('iÎ,Îa->ia', I_iÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiia += np.einsum('ia,bacd->cdib', I_ia, t_aaii, optimize=True)
    del t_aaii
    del I_ia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->aib', I2_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I2_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iÎ += np.einsum('iaÎ,abÎ->ib', I_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iiÎ
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_ii += np.einsum('iÎ,aÎ->ai', I_iÎ, I2_iÎ, optimize=True)
    del I2_iÎ
    del I_iÎ
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I2_iiaa += np.einsum('ia,bc->iabc', I_ii, I_aa, optimize=True)
    del I_aa
    del I_ii
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += 2 * np.einsum('iabc,daec->dieb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎ = np.zeros((dim_i, dim_i, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎ
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎ
    I_iiiÎ += np.einsum('iaÎ,bcÎ->biac', I_iiÎ, I_iÎÎ, optimize=True)
    del I_iÎÎ
    del I_iiÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiia += np.einsum('iabÎ,cÎ->iabc', I_iiiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiiÎ
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += -2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdea->eibd', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iadc', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdae->eibd', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiii = np.zeros((dim_i, dim_i, dim_i, dim_i), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_iiii += np.einsum('iabÎ,cÎ->ciab', I_iiiÎ, I_iÎ, optimize=True)
    del I_iÎ
    del I_iiiÎ
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiia += np.einsum('iabc,cd->iabd', I_iiii, I_ia, optimize=True)
    del I_ia
    del I_iiii
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdbe->eiad', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += -4 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I3_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiia += np.einsum('iabÎ,Îc->iabc', I_iiiÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiiÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiia += np.einsum('iabc,cdeb->eiad', I3_iiia, t_aaii, optimize=True)
    del t_aaii
    del I3_iiia
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiia += np.einsum('iabc,dc->iabd', I2_iiia, I_aa, optimize=True)
    del I_aa
    del I2_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += 2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iiÎÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del g_iiÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_iiiÎ += np.einsum('iaÎb,cb->ciaÎ', I_iiÎÎ, I_iÎ, optimize=True)
    del I_iÎ
    del I_iiÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiia += np.einsum('iabÎ,cÎ->iabc', I_iiiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiiÎ
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiaa += -2 * np.einsum('iabc,bd->iacd', I_iiia, I_ia, optimize=True)
    del I_ia
    del I_iiia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I2_ia = np.zeros((dim_i, dim_a), order='F')
    f_iÎ = np.load('f_iÎ.npy')
    C_Îa = np.load('C_Îa.npy')
    I2_ia += np.einsum('iÎ,Îa->ia', f_iÎ, C_Îa, optimize=True)
    del C_Îa
    del f_iÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiia += np.einsum('ia,bacd->cdib', I2_ia, t_aaii, optimize=True)
    del t_aaii
    del I2_ia
    I_iiaa += -2 * np.einsum('ia,bcid->bcda', I_ia, I_iiia, optimize=True)
    del I_iiia
    del I_ia
    I_ia = np.zeros((dim_i, dim_a), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    I2_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I2_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    s_ÎÎ = np.load('s_ÎÎ.npy')
    I_iÎ += np.einsum('iÎ,aÎ->ia', I2_iÎ, s_ÎÎ, optimize=True)
    del s_ÎÎ
    del I2_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_ia += np.einsum('iÎ,aÎ->ia', I_iÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎ
    I_iiia = np.zeros((dim_i, dim_i, dim_i, dim_a), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_ÎiÎ = np.load('g_ÎiÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->caib', g_iiÎ, g_ÎiÎ, optimize=True)
    del g_ÎiÎ
    del g_iiÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiia += np.einsum('iabÎ,cÎ->iabc', I_iiiÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiiÎ
    I_iiaa += -2 * np.einsum('ia,bcid->bcda', I_ia, I_iiia, optimize=True)
    del I_iiia
    del I_ia
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I_iiiÎ = np.zeros((dim_i, dim_i, dim_i, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_iiiÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del g_iiÎ
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    I_ii += np.einsum('iabÎ,aÎ->ib', I_iiiÎ, I_iÎ, optimize=True)
    del I_iÎ
    del I_iiiÎ
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I2_iiaa += np.einsum('ia,bc->iabc', I_ii, I_aa, optimize=True)
    del I_aa
    del I_ii
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += 2 * np.einsum('iabc,daec->dieb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
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
    I3_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    I4_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I4_iiaa += np.einsum('ab,cbid->diac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I2_iiaa += np.einsum('iabc,daeb->diec', I3_iiaa, I4_iiaa, optimize=True)
    del I4_iiaa
    del I3_iiaa
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,cbid->diac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += -2 * np.einsum('iabc,daec->dieb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
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
    I3_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    t_aaii = np.load('t_aaii.npy')
    I_ii += np.einsum('iabc,bcda->di', I3_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I3_iiaa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I2_iiaa += np.einsum('ia,bc->iabc', I_ii, I_aa, optimize=True)
    del I_aa
    del I_ii
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += -4 * np.einsum('iabc,daec->dieb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_ii = np.zeros((dim_i, dim_i), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
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
    I3_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    t_aaii = np.load('t_aaii.npy')
    I_ii += np.einsum('iabc,bcdi->da', I3_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I3_iiaa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I2_iiaa += np.einsum('ia,bc->iabc', I_ii, I_aa, optimize=True)
    del I_aa
    del I_ii
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += 2 * np.einsum('iabc,daec->dieb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    f_ii = np.load('f_ii.npy')
    I2_iiaa += np.einsum('ab,ic->ciab', I_aa, f_ii, optimize=True)
    del f_ii
    del I_aa
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I3_iiaa += np.einsum('ab,bcid->idac', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaa += -2 * np.einsum('iabc,daec->dieb', I2_iiaa, I3_iiaa, optimize=True)
    del I3_iiaa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iiÎÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del g_iiÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiaÎ += np.einsum('iaÎb,cÎ->iacb', I_iiÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiÎÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiaa += np.einsum('iabc,cdea->iebd', I3_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I3_iiaa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiaa += -2 * np.einsum('iabc,dc->iadb', I2_iiaa, I_aa, optimize=True)
    del I_aa
    del I2_iiaa
    I2_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I3_iiaa = np.zeros((dim_i, dim_i, dim_a, dim_a), order='F')
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iiÎÎ = np.zeros((dim_i, dim_i, dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    g_iiÎ = np.load('g_iiÎ.npy')
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iiÎÎ += np.einsum('iaÎ,bcÎ->aibc', g_iiÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del g_iiÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiaÎ += np.einsum('iaÎb,cÎ->iacb', I_iiÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiÎÎ
    C_Îa = np.load('C_Îa.npy')
    I3_iiaa += np.einsum('iabÎ,Îc->iabc', I_iiaÎ, C_Îa, optimize=True)
    del C_Îa
    del I_iiaÎ
    t_aaii = np.load('t_aaii.npy')
    I2_iiaa += np.einsum('iabc,cdae->iebd', I3_iiaa, t_aaii, optimize=True)
    del t_aaii
    del I3_iiaa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    s_ÎÎ = np.load('s_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', s_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del s_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    I_iiaa += -2 * np.einsum('iabc,dc->iabd', I2_iiaa, I_aa, optimize=True)
    del I_aa
    del I2_iiaa
    I_iiaÎ = np.zeros((dim_i, dim_i, dim_a, dim_Î¼Ìƒ), order='F')
    I_iaÎ = np.zeros((dim_i, dim_a, dim_Îš), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iaÎ += np.einsum('iÎa,bÎ->iba', I_iÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iÎÎ
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎ
    I_iiaÎ += np.einsum('iaÎ,bcÎ->ibac', I_iaÎ, I_iÎÎ, optimize=True)
    del I_iÎÎ
    del I_iaÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_iiaa += np.einsum('iabÎ,cÎ->iabc', I_iiaÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_iiaÎ
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    I_ÎÎ = np.zeros((dim_Î¼Ìƒ, dim_Î¼Ìƒ), order='F')
    I_iÎÎ = np.zeros((dim_i, dim_Î¼Ìƒ, dim_Îš), order='F')
    I_iÎ = np.zeros((dim_i, dim_Î¼Ìƒ), order='F')
    C_Îa = np.load('C_Îa.npy')
    t_ai = np.load('t_ai.npy')
    I_iÎ += np.einsum('Îa,ai->iÎ', C_Îa, t_ai, optimize=True)
    del t_ai
    del C_Îa
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    I_iÎÎ += np.einsum('iÎ,aÎb->iab', I_iÎ, g_ÎÎÎ, optimize=True)
    del g_ÎÎÎ
    del I_iÎ
    g_iÎÎ = np.load('g_iÎÎ.npy')
    I_ÎÎ += np.einsum('iÎa,iba->bÎ', I_iÎÎ, g_iÎÎ, optimize=True)
    del g_iÎÎ
    del I_iÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aÎ += np.einsum('Îa,Îb->ba', I_ÎÎ, C_Îa, optimize=True)
    del C_Îa
    del I_ÎÎ
    C_aÎ = np.load('C_aÎ.npy')
    I_aa += np.einsum('aÎ,bÎ->ba', I_aÎ, C_aÎ, optimize=True)
    del C_aÎ
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiaa += -2 * np.einsum('ab,cbid->idca', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_aa = np.zeros((dim_a, dim_a), order='F')
    I_aÎ = np.zeros((dim_a, dim_Î¼Ìƒ), order='F')
    f_ÎÎ = np.load('f_ÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎ += np.einsum('Îa,bÎ->ba', f_ÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del f_ÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aa += np.einsum('aÎ,Îb->ab', I_aÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiaa += 2 * np.einsum('ab,cbid->idca', I_aa, t_aaii, optimize=True)
    del t_aaii
    del I_aa
    I_iiaaÎ = np.zeros((dim_i, dim_i, dim_a, dim_a, dim_Îš), order='F')
    I_aaÎ = np.zeros((dim_a, dim_a, dim_Îš), order='F')
    I_aÎÎ = np.zeros((dim_a, dim_Î¼Ìƒ, dim_Îš), order='F')
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎÎ += np.einsum('Îab,cÎ->cab', g_ÎÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del g_ÎÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aaÎ += np.einsum('aÎb,Îc->acb', I_aÎÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎÎ
    t_aaii = np.load('t_aaii.npy')
    I_iiaaÎ += np.einsum('abÎ,bcid->idacÎ', I_aaÎ, t_aaii, optimize=True)
    del t_aaii
    del I_aaÎ
    I_aaÎ = np.zeros((dim_a, dim_a, dim_Îš), order='F')
    I_aÎÎ = np.zeros((dim_a, dim_Î¼Ìƒ, dim_Îš), order='F')
    g_ÎÎÎ = np.load('g_ÎÎÎ.npy')
    C_aÎ = np.load('C_aÎ.npy')
    I_aÎÎ += np.einsum('Îab,cÎ->cab', g_ÎÎÎ, C_aÎ, optimize=True)
    del C_aÎ
    del g_ÎÎÎ
    C_Îa = np.load('C_Îa.npy')
    I_aaÎ += np.einsum('aÎb,Îc->acb', I_aÎÎ, C_Îa, optimize=True)
    del C_Îa
    del I_aÎÎ
    I_iiaa += np.einsum('iabcÎ,dcÎ->iabd', I_iiaaÎ, I_aaÎ, optimize=True)
    del I_aaÎ
    del I_iiaaÎ
    np.save('I_iiaa.npy', I_iiaa)


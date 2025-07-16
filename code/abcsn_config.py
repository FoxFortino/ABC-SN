import numpy as np

FILT = True
FILT = False
SNIDIFIED = False
SNIDIFIED = True

default_hyper_parameters = {
    "phase_range": (-20, 50),
    "ptp_range": (0.1, 100),
    "wvl_range": (4500, 7000),
    "train_frac": 0.50,
    "noise_scale": 0.25,
    "spike_scale": 3,
    "max_spikes": 5,
    "random_state": 1415,

    "lr0": 0.001,
    "lr_schedule": "constant_lr",

    "num_transformer_blocks": 6,
    "num_heads": 8,
    "key_dim": 64,
    "kr_l2": 0,
    "br_l2": 0,
    "ar_l2": 0,
    "dropout_attention": 0.1,
    "dropout_projection": 0.1,
    "filters": 4,
    "num_feed_forward_layers": 1,
    "feed_forward_layer_size": 1024,
    "dropout_feed_forward": 0.1,
}


def determine_ID_Stype_to_Mtype(SN_Stype_ID: int):
    """
    Given an SN subtype ID, return the maintype ID.
    """
    if 0 <= SN_Stype_ID <= 5:
        return 0
    elif 6 <= SN_Stype_ID <= 9:
        return 1
    elif 10 <= SN_Stype_ID <= 12:
        return 2
    elif 13 <= SN_Stype_ID <= 16:
        return 3
    else:
        raise ValueError("SN_Stype_ID should be between 0 and 16 "
                         f"(inclusive), but was {SN_Stype_ID}.")
determine_ID_Stype_to_Mtype = np.vectorize(determine_ID_Stype_to_Mtype,
                                           signature="()->()")


def get_Mtype_str_from_ID(SN_Mtype_ID: int):
    """
    Given an SN maintype ID, return the string that corresponds to that type.
    """
    if SN_Mtype_ID == 0:
        return "Ia"
    elif SN_Mtype_ID == 1:
        return "Ib"
    elif SN_Mtype_ID == 2:
        return "Ic"
    elif SN_Mtype_ID == 3:
        return "II"
    else:
        raise ValueError("SN_Mtype_ID should be 0, 1, 2, or 3 but was "
                         f"{SN_Mtype_ID}.")
get_Mtype_str_from_ID = np.vectorize(get_Mtype_str_from_ID,
                                     signature="()->()")


def get_Mtype_ID_from_str(SN_Mtype: str):
    """
    Given an SN maintype string, return the ID that corresponds to that type.
    """
    if SN_Mtype == "Ia":
        return 0
    elif SN_Mtype == "Ib":
        return 1
    elif SN_Mtype == "Ic":
        return 2
    elif SN_Mtype == "II":
        return 3
    else:
        raise ValueError("SN_Mtype should be 'Ia', 'Ib', 'Ic', or 'II' but "
                         f"was {SN_Mtype}.")
get_Mtype_ID_from_str = np.vectorize(get_Mtype_ID_from_str,
                                     signature="()->()")

# 17 SN sub-types
SN_Stypes_str = np.array(
    ["Ia-norm", "Ia-91T", "Ia-91bg", "Ia-csm", "Iax", "Ia-pec",
     "Ib-norm", "Ibn", "IIb", "Ib-pec",
     "Ic-norm", "Ic-broad", "Ic-pec",
     "IIP", "IIL", "IIn", "II-pec"])
SN_Stypes_int = np.arange(SN_Stypes_str.size)
SN_Stypes_str_to_int = {s: i for s, i in zip(SN_Stypes_str, SN_Stypes_int)}
SN_Stypes_int_to_str = {i: s for s, i in SN_Stypes_str_to_int.items()}

# 4 SN main-types
SN_Mtypes_str = np.array(["Ia", "Ib", "Ic", "II"])
SN_Mtypes_int = np.arange(SN_Mtypes_str.size)
SN_Mtypes_str_to_int = {s: i for s, i in zip(SN_Mtypes_str, SN_Mtypes_int)}
SN_Mtypes_int_to_str = {j: i for i, j in SN_Mtypes_str_to_int.items()}


"""Common corrections for SN names"""
SN_Stypes_str_to_int["Ia-02cx"] = SN_Stypes_str_to_int["Iax"]
SN_Stypes_str_to_int["Ia-99aa"] = SN_Stypes_str_to_int["Ia-91T"]
SN_Stypes_str_to_int["Ib"] = SN_Stypes_str_to_int["Ib-norm"]
SN_Stypes_str_to_int["Ic"] = SN_Stypes_str_to_int["Ic-norm"]

# The following four hardcoded lists of supernova were taken from
# create_templist.py form the astrodash GitHub repository.

# WWW Run this list by Somayeh she might have more info on these
# Delete Files from templates-2.0 and Liu & Modjaz
NO_MAX_SNID_LIU_MODJAZ = [
    "sn1997X", "sn2001ai", "sn2001ej", "sn2001gd","sn2001ig", "sn2002ji",
    "sn2004ao", "sn2004eu", "sn2004gk", "sn2005ar", "sn2005da", "sn2005kf",
    "sn2005nb", "sn2005U", "sn2006ck", "sn2006fo", "sn2006lc", "sn2006lv",
    "sn2006ld", "sn2007ce", "sn2007I", "sn2007rz", "sn2008an", "sn2008aq",
    "sn2008cw", "sn1988L", "sn1990K", "sn1990aa", "sn1991A", "sn1991N",
    "sn1991ar", "sn1995F", "sn1997cy", "sn1997dc", "sn1997dd", "sn1997dq",
    "sn1997ei", "sn1998T", "sn1999di", "sn1999dn", "sn2004dj"
]

BAD_SPECTRA = ['sn2010bh']

# Delete files from bsnip
NO_MAX_BSNIP_AGE_999 = [
    'sn00ev_bsnip.lnw', 'sn00fe_bsnip.lnw', 'sn01ad_bsnip.lnw',
    'sn01cm_bsnip.lnw', 'sn01cy_bsnip.lnw', 'sn01dk_bsnip.lnw',
    'sn01do_bsnip.lnw', 'sn01ef_bsnip.lnw', 'sn01ey_bsnip.lnw',
    'sn01gd_bsnip.lnw', 'sn01hg_bsnip.lnw', 'sn01ir_bsnip.lnw',
    'sn01K_bsnip.lnw', 'sn01M_bsnip.lnw', 'sn01X_bsnip.lnw',
    'sn02A_bsnip.lnw', 'sn02an_bsnip.lnw', 'sn02ap_bsnip.lnw',
    'sn02bu_bsnip.lnw', 'sn02bx_bsnip.lnw', 'sn02ca_bsnip.lnw',
    'sn02dq_bsnip.lnw', 'sn02eg_bsnip.lnw', 'sn02ei_bsnip.lnw',
    'sn02eo_bsnip.lnw', 'sn02hk_bsnip.lnw', 'sn02hn_bsnip.lnw',
    'sn02J_bsnip.lnw', 'sn02kg_bsnip.lnw', 'sn03ab_bsnip.lnw',
    'sn03B_bsnip.lnw', 'sn03ei_bsnip.lnw', 'sn03G_bsnip.lnw',
    'sn03gd_bsnip.lnw', 'sn03gg_bsnip.lnw', 'sn03gu_bsnip.lnw',
    'sn03hl_bsnip.lnw', 'sn03ip_bsnip.lnw', 'sn03iq_bsnip.lnw',
    'sn03kb_bsnip.lnw', 'sn04aq_bsnip.lnw', 'sn04bi_bsnip.lnw',
    'sn04cz_bsnip.lnw', 'sn04dd_bsnip.lnw', 'sn04dj_bsnip.lnw',
    'sn04du_bsnip.lnw', 'sn04et_bsnip.lnw', 'sn04eu_bsnip.lnw',
    'sn04ez_bsnip.lnw', 'sn04fc_bsnip.lnw', 'sn04fx_bsnip.lnw',
    'sn04gd_bsnip.lnw', 'sn04gr_bsnip.lnw', 'sn05ad_bsnip.lnw',
    'sn05af_bsnip.lnw', 'sn05aq_bsnip.lnw', 'sn05ay_bsnip.lnw',
    'sn05bi_bsnip.lnw', 'sn05bx_bsnip.lnw', 'sn05cs_bsnip.lnw',
    'sn05ip_bsnip.lnw', 'sn05kd_bsnip.lnw', 'sn06ab_bsnip.lnw',
    'sn06be_bsnip.lnw', 'sn06bp_bsnip.lnw', 'sn06by_bsnip.lnw',
    'sn06ca_bsnip.lnw', 'sn06cx_bsnip.lnw', 'sn06gy_bsnip.lnw',
    'sn06my_bsnip.lnw', 'sn06ov_bsnip.lnw', 'sn06T_bsnip.lnw',
    'sn06tf_bsnip.lnw', 'sn07aa_bsnip.lnw', 'sn07ag_bsnip.lnw',
    'sn07av_bsnip.lnw', 'sn07ay_bsnip.lnw', 'sn07bb_bsnip.lnw',
    'sn07be_bsnip.lnw', 'sn07C_bsnip.lnw', 'sn07ck_bsnip.lnw',
    'sn07cl_bsnip.lnw', 'sn07K_bsnip.lnw', 'sn07oc_bsnip.lnw',
    'sn07od_bsnip.lnw', 'sn08aq_bsnip.lnw', 'sn08aw_bsnip.lnw',
    'sn08be_bsnip.lnw', 'sn08bj_bsnip.lnw', 'sn08bl_bsnip.lnw',
    'sn08D_bsnip.lnw', 'sn08es_bsnip.lnw', 'sn08fq_bsnip.lnw',
    'sn08gf_bsnip.lnw', 'sn08gj_bsnip.lnw', 'sn08ht_bsnip.lnw',
    'sn08in_bsnip.lnw', 'sn08iy_bsnip.lnw', 'sn88Z_bsnip.lnw',
    'sn90H_bsnip.lnw', 'sn90Q_bsnip.lnw', 'sn91ao_bsnip.lnw',
    'sn91av_bsnip.lnw', 'sn91C_bsnip.lnw', 'sn92ad_bsnip.lnw',
    'sn92H_bsnip.lnw', 'sn93ad_bsnip.lnw', 'sn93E_bsnip.lnw',
    'sn93G_bsnip.lnw', 'sn93J_bsnip.lnw', 'sn93W_bsnip.lnw',
    'sn94ak_bsnip.lnw', 'sn94I_bsnip.lnw', 'sn94W_bsnip.lnw',
    'sn94Y_bsnip.lnw', 'sn95G_bsnip.lnw', 'sn95J_bsnip.lnw',
    'sn95V_bsnip.lnw', 'sn95X_bsnip.lnw', 'sn96ae_bsnip.lnw',
    'sn96an_bsnip.lnw', 'sn96cc_bsnip.lnw', 'sn97ab_bsnip.lnw',
    'sn97da_bsnip.lnw', 'sn97dd_bsnip.lnw', 'sn97ef_bsnip.lnw',
    'sn97eg_bsnip.lnw', 'sn98A_bsnip.lnw', 'sn98dl_bsnip.lnw',
    'sn98dt_bsnip.lnw', 'sn98E_bsnip.lnw', 'sn98S_bsnip.lnw',
    'sn99eb_bsnip.lnw', 'sn99ed_bsnip.lnw', 'sn99el_bsnip.lnw',
    'sn99em_bsnip.lnw', 'sn99gb_bsnip.lnw', 'sn99gi_bsnip.lnw',
    'sn99Z_bsnip.lnw'
]

SAME_SN_WITH_SAME_AGES_AS_SNID = [
    'sn02ic.lnw', 'sn04dj.lnw', 'sn05gj.lnw', 'sn05hk.lnw',
    'sn96L.lnw', 'sn99ex.lnw', 'sn02ap.lnw', 'sn02bo.lnw',
    'sn04aw_bsnip.lnw', 'sn04et.lnw', 'sn05cs.lnw', 'sn90N.lnw',
    'sn92A.lnw', 'sn93J.lnw', 'sn97br.lnw', 'sn97ef.lnw',
    'sn98S.lnw', 'sn99aa.lnw', 'sn99em.lnw'
]

WFF_WEIRD_SN = [
    "12au.lnw"
]

ALL_BAD_SN = []
ALL_BAD_SN += NO_MAX_SNID_LIU_MODJAZ
ALL_BAD_SN += BAD_SPECTRA
ALL_BAD_SN += NO_MAX_BSNIP_AGE_999
ALL_BAD_SN += SAME_SN_WITH_SAME_AGES_AS_SNID
# ALL_BAD_SN += WFF_WEIRD_SN

ALL_BAD_SN = [sn.split(".lnw")[0] for sn in ALL_BAD_SN]

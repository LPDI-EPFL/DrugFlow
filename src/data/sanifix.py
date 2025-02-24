""" sanifix4.py

  Contribution from James Davidson
  adapted from: https://github.com/abradle/rdkitserver/blob/master/MYSITE/src/testproject/mol_parsing/sanifix.py
"""
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings

def _FragIndicesToMol(oMol,indices):
    em = Chem.EditableMol(Chem.Mol())

    newIndices={}
    for i,idx in enumerate(indices):
        em.AddAtom(oMol.GetAtomWithIdx(idx))
        newIndices[idx]=i

    for i,idx in enumerate(indices):
        at = oMol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx()==idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once:
            if oidx<idx:
                continue
            em.AddBond(newIndices[idx],newIndices[oidx],bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    Chem.GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    res._idxMap=newIndices
    return res

def _recursivelyModifyNs(mol,matches,indices=None):
    if indices is None:
        indices=[]
    res=None
    while len(matches) and res is None:
        tIndices=indices[:]
        nextIdx = matches.pop(0)
        tIndices.append(nextIdx)
        nm = Chem.Mol(mol)
        nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
        nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
        cp = Chem.Mol(nm)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            res,indices = _recursivelyModifyNs(nm,matches,indices=tIndices)
        else:
            indices=tIndices
            res=cp
    return res,indices

def AdjustAromaticNs(m,nitrogenPattern='[n&D2&H0;r5,r6]'):
    """
       default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
       to fix: O=c1ccncc1
    """
    Chem.GetSymmSSSR(m)
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings:
    em = Chem.EditableMol(m)
    linkers = m.GetSubstructMatches(Chem.MolFromSmarts('[r]!@[r]'))
    plsFix=set()
    for a,b in linkers:
        em.RemoveBond(a,b)
        plsFix.add(a)
        plsFix.add(b)
    nm = em.GetMol()
    for at in plsFix:
        at=nm.GetAtomWithIdx(at)
        if at.GetIsAromatic() and at.GetAtomicNum()==7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments:
    fragLists = Chem.GetMolFrags(nm)
    frags = [_FragIndicesToMol(nm,x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    ok=True
    for i,frag in enumerate(frags):
        cp = Chem.Mol(frag)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            matches = [x[0] for x in frag.GetSubstructMatches(Chem.MolFromSmarts(nitrogenPattern))]
            lres,indices=_recursivelyModifyNs(frag,matches)
            if not lres:
                #print 'frag %d failed (%s)'%(i,str(fragLists[i]))
                ok=False
                break
            else:
                revMap={}
                for k,v in frag._idxMap.items():
                    revMap[v]=k
                for idx in indices:
                    oatom = m.GetAtomWithIdx(revMap[idx])
                    oatom.SetNoImplicit(True)
                    oatom.SetNumExplicitHs(1)
    if not ok:
        return None
    return m



def fix_mol(m):
    if m is None:
        return None
    try:
        m.UpdatePropertyCache(False)
        cp = Chem.Mol(m.ToBinary())
        Chem.SanitizeMol(cp)
        m = cp
        # print('fine:',Chem.MolToSmiles(m))
        warnings.warn(f'fine: {Chem.MolToSmiles(m)}')
        return m
    except ValueError:
        # print('adjust')
        warnings.warn('adjust')
        nm=AdjustAromaticNs(m)
        if nm is not None:
            try:
                Chem.SanitizeMol(nm)
                # print('fixed:',Chem.MolToSmiles(nm))
                warnings.warn(f'fixed: {Chem.MolToSmiles(nm)}')
            except ValueError:
                # print('still broken')
                warnings.warn('still broken')
        else:
            # print('still broken')
            warnings.warn('still broken')
        return nm

if __name__=='__main__':
    ms = [x for x in open("18.sdf").read().split("$$$$\n")]
    for txt_m in ms:
        if not txt_m:
            continue
        m = Chem.MolFromMolBlock(txt_m, False)
        print('#---------------------')
        try:
            m.UpdatePropertyCache(False)
            cp = Chem.Mol(m.ToBinary())
            Chem.SanitizeMol(cp)
            m = cp
            print('fine:',Chem.MolToSmiles(m))
        except ValueError:
            print('adjust')
            nm=AdjustAromaticNs(m)
            if nm is not None:
                Chem.SanitizeMol(nm)
                print('fixed:',Chem.MolToSmiles(nm))
            else:
                print('still broken')

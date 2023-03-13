"""  Created on 17/09/2021::
------------- commonprob.py -------------

**Authors**: L. Mingarelli
"""
import numpy as np
import warnings

def check_commonprob(commonprob):
    """
    Tests admissibility of the joint probability matrix.

    The main diagonal elements commonprob[i,i] are interpreted as probabilities p_Ai
    that a binary variable Ai equals 1.
    The off-diagonal elements commonprob[i,j] are the probabilities p_AiAj
    that both Ai and Aj are 1.
    This function checks some necessary conditions on these probabilities which must be fulfilled in
    order for a joint distribution to exist.
    The conditions checked are:

            0 ≤ p_Ai ≤ 1
            max(0, p_Ai + p_Aj − 1) ≤ p_AiAj ≤ min(p_Ai, p_Aj), i ≠ j
            p_Ai + p_Aj + p_Ak − p_AiAj − p_AiAk − p_AjAk ≤ 1, i ≠ j, i ≠ k, j ≠ k

    Args:
        commonprob (numpy.array): the joint_probability matrix
    Returns:
        bool, message

    """
    commonprob = np.array(commonprob)
    flag = True
    msg = []

    if np.any(commonprob < 0) or np.any(commonprob > 1): # Check for entris outside of unit interval
        flag = False
        msg.append("Not all probabilities are between 0 and 1.")

    n = commonprob.shape[0]
    if n != commonprob.shape[1]: # Check for non-square matrix
        flag = False
        msg.append("Matrix of common probabilities is not square.")

    # Check pairwise conditions
    i, j = np.triu_indices(n, k=1)
    ul = np.minimum(commonprob[i, i], commonprob[j, j])
    ll = np.maximum(commonprob[i, i] + commonprob[j, j] - 1, 0)
    invalid_pairs = np.where((commonprob[i, j] > ul) | (commonprob[i, j] < ll))
    if invalid_pairs[0].size > 0:
        invalid_pairs = np.column_stack((i[invalid_pairs], j[invalid_pairs]))
        message = [f"Error in Element ({p[0]}, {p[1]}): Admissible values are in [{round(ll[k], 10)}, {round(ul[k], 10)}]."
                   for k, p in enumerate(invalid_pairs)]
        flag = False
        msg += message

    # check triple conditions
    if n > 2:
        i, j, k = np.array([(i, j, k)
                            for i in range(n - 2)
                            for j in range(i + 1, n - 1)
                            for k in range(j + 1, n)]).T
        l = commonprob[i, i] + commonprob[j, j] + commonprob[k, k] - 1
        invalid_triples = np.where(commonprob[i, j] + commonprob[i, k] + commonprob[j, k] < l)[0]
        if invalid_triples.size > 0:
            invalid_triples = np.column_stack((i[invalid_triples],
                                               j[invalid_triples],
                                               k[invalid_triples]))
            message = [f"The sum of the common probabilities of {p[0]}, {p[1]}, {p[2]} must be at least {l[k]}."
                       for k, p in enumerate(invalid_triples)]
            flag = False
            msg += message

    # Input satisfies all conditions
    return flag, msg


def _check_against_simulvals(x, simulvals):
    non_computable_entries = ~np.isin(x.round(10),
                                      np.array(list(simulvals.keys())))
    if non_computable_entries.any():
        warnings.warn(f"simulvals provided is not computed to a sufficient resolution\n"
                         f"to resolve the common probabilies provided in commonprob.\n"
                         f"\n\nINTERPOLATION IS GOING TO BE USED.\n\n"
                         f"Consider computing simulvals on a finer grid.\n"
                         f"The current resolution is on these points:\n{list(simulvals.keys())}\n"
                         f"Alternatively round the common probabilities or correlations provided\n"
                         f"to fewer decimal places."
                         )




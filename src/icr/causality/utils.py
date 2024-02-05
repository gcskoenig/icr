from icr.causality.scms import BinomialBinarySCM, GenericSCM

def indvd_to_intrv(scm, features, individual, obs, causes_of=None):
    """
    If causes_of is None, then all interventions are added to the dictionary.
    If causes of is specified, only internvetions on ancestors of the specified
    node are considered.
    """
    dict = {}

    # build causes set
    causes = None
    if causes_of is None:
        causes = set(features)
    else:
        causes = scm.dag.get_ancestors_node(causes_of)

    # iterate over variables to add causes
    for ii in range(len(features)):
        var_name = features[ii]
        if abs(individual[ii]) > 0 and (var_name in causes):
            if isinstance(scm, BinomialBinarySCM):
                dict[var_name] = (obs[var_name] + individual[ii]) % 2
            elif isinstance(scm, GenericSCM):
                dict[var_name] = obs[var_name] + individual[ii]
            else:
                raise NotImplementedError('only BinomialBinary or GenericSCM supported.')
    return dict
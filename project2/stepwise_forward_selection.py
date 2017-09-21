""" Created by Max Robinson 9/20/2017 """


class SFS:
    pass


"""
Pseudo code

function SFS(Features, D_train, D_valid, Learn()): 
    F_0 = <>
    basePerf = -inf
    do:
        bestPerf = - inf
        for all Features in FeatureSpace do: 
            F_0 = F_0 + F
            h = Learn(F_0, D_train)
            currPerf = Perf(h, D_valid)
            if currPerf > bestPerf then:
                bestPerf = currPerf
                bestF = F
            end if
            F_0 = F_0 - F
        end for
        if bestPerf > basePerf then 
            basePerf = bestPerf
            F = F - bestF 
            F_0 = F_0 + bestF
        else
            exit (Break)
        end if
    until F = <> (is empty)
    return F_0

"""
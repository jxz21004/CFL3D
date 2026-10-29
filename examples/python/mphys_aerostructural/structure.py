"""Same isotropic shell and gravity inputs as the supplied L2 TACS callback."""
import numpy as np
from tacs import constitutive, elements, functions


def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    if any(name != "CQUAD4" for name in elemDescripts):
        raise ValueError(f"This wingbox example supports CQUAD4, got {elemDescripts}")
    prop = constitutive.MaterialProperties(rho=2500.0, E=70e9, nu=0.3, ys=350e6)
    con = constitutive.IsoShellConstitutive(prop, t=0.01, tNum=dvNum, tlb=0.002, tub=0.05)
    return elements.Quad4Shell(None, con)


def problem_setup(directory):
    def setup(name, assembler, problem):
        problem.setOptions({"L2ConvergenceRel": 1e-12, "L2Convergence": 1e-12,
                            "outputDir": str(directory)})
        if name == "cruise":
            problem.addFunction("mass", functions.StructuralMass)
        problem.addFunction("ks_vmfailure", functions.KSFailure, safetyFactor=1.0, ksWeight=100.0)
        problem.addInertialLoad(np.array([0., 0., -9.81]) * (2.5 if name == "maneuver" else 1.0))
    return setup

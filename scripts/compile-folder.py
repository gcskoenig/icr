from icr.experiment.compile import compile_experiments
from icr.causality.scms.examples import scm_dict
import argparse
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile experiments")

    parser.add_argument("scm_name", help=f"one of {scm_dict.keys()}", type=str)
    parser.add_argument("savepath",
                        help="savepath for the experiment folder. either relative to working directory or absolute.",
                        type=str)

    parser.add_argument("--ignore_np_errs", type=bool, default=True)
    parser.add_argument("--logging-level", type=int, default=20)

    args = parser.parse_args()

    # set logging settings
    logging.getLogger().setLevel(args.logging_level)

    if args.ignore_np_errs:
        np.seterr(all="ignore")

    compile_experiments(args.savepath, args.scm_name)
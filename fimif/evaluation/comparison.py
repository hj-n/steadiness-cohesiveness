"""
This will compare embedding result quantitatively using functions in utils.py
"""

# from .models.dataset import read_data
from utils import GlobalMeasure, LocalMeasure
import pandas as pd
import argparse
import numpy as np
import json

MEASURE_GLOBAL_LIST = [
    # "RMSE",
    # "Kruskal",
    # "Sammon",
    "DTM",
    "DTM_KL1",
    "DTM_KL01",
    "DTM_KL001",
]

MEASURE_LOCAL_LIST = [
    # "Spearman",
    "Trustworthiness",
    "Continuity",
    "MRRE_XZ",
    "MRRE_ZX",
]

ALGO_LIST = ["pca", "tsne", "umap", "topoae", "atsne", "umato"]
# ALGO_LIST = ["topoae", "atsne"]
DATA_LIST = ["spheres"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="quantitative comparison of the embedding result"
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="choose algorithm: pca, tsne, umap, topoae, atsne, umato",
        default="all",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="choose dataset: spheres, mnist, fmnist, cifar10",
        default="spheres",
    )
    parser.add_argument(
        "--measure", type=str, help="choose measures: all, global, local", default="all"
    )
    parser.add_argument(
        "--k", type=int, help="number of nearest neighbors", default=5
    )
    parser.add_argument(
        "--load", type=bool, help="load hubs", default=False
    )
    args = parser.parse_args()

    measures = []
    algorithms = []
    values = []

    if args.algo != "all":
        ALGO_LIST = [args.algo]
        
    if args.algo == "ss":  ## swiss roll
        ALGO_LIST = []
        ## multiclass swissroll
        # for i in range(-7, 8):
        #     ALGO_LIST.append("multiclass_swissroll_" + str(i) + "_none")
        # for i in range(0, 14):
        #     ALGO_LIST.append("multiclass_swissroll_oneside_" + str(i) + "_none")
        for i in range(0, 15):
            ALGO_LIST.append("multiclass_swissroll_half_" + str(i) + "_none")
    
    if args.algo == "mnist":
        ALGO_LIST = []
        for i in [1, 5, 50, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]:
            ALGO_LIST.append("mnist_test_" + str(i) + "_tsne")
    
    if args.algo == "mammoth":
        ALGO_LIST = [] 
        for n in [3, 5, 10, 15, 20, 50, 100,200]:
            for d in [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]:
                key_summary = str(n) + "_" + str(d)
                ALGO_LIST.append("mammoth_" + key_summary)

    if args.algo == "spheres":
        ALGO_LIST = []
        ALGO_LIST = [] 
        for n in [3, 10, 20, 30, 40, 50, 100, 150, 200, 400, 600, 800, 1000]:
            for d in [0.0, 0.2, 0.4, 0.5, 0.99]:
                key_summary = str(n) + "_" + str(d)
                ALGO_LIST.append("spheres_sampled_" + key_summary)

    for alg in ALGO_LIST:

        # print(f"[INFO] Test on [{args.data}] dataset using [{alg}]")

        # read data & embedding result
        # x, z, label = read_data(args.data, alg)
        json_file = open("../measure/json/" + alg + ".json", "r") 
        json_data = json.load(json_file)
        x = np.array([datum["raw"] for datum in json_data]).astype(np.float64)
        z = np.array([datum["emb"] for datum in json_data]).astype(np.float64)
        label = np.array([datum["label"] for datum in json_data]).astype(np.float64)

        if args.load:
            with open('./hubs.npy', 'rb') as f:
                print("loading hubs")
                hubs = np.load(f)
                x = x[hubs]
                z = z[hubs]
                print(f"xlen: {len(x)}")

        if args.measure == "all" or args.measure == "global":

            gmeasure = GlobalMeasure(x, z)

            algorithms.extend([alg] * len(MEASURE_GLOBAL_LIST))
            measures.extend(MEASURE_GLOBAL_LIST)

            # rmse_val = gmeasure.rmse()
            # kruskal_val = gmeasure.kruskal_stress_measure()
            # sammon_val = gmeasure.sammon_stress()
            dtm_val = gmeasure.dtm()
            dtmkl1_val = gmeasure.dtm_kl(sigma=1.0)
            dtmkl01_val = gmeasure.dtm_kl(sigma=0.1)
            dtmkl001_val = gmeasure.dtm_kl(sigma=0.01)
            values.extend(
                [
                    # rmse_val,
                    # kruskal_val,
                    # sammon_val,
                    dtm_val,
                    dtmkl1_val,
                    dtmkl01_val,
                    dtmkl001_val,
                ]
            )

        if args.measure == "all" or args.measure == "local":

            lmeasure = LocalMeasure(x, z, k=args.k)

            algorithms.extend([alg] * len(MEASURE_LOCAL_LIST))
            measures.extend(MEASURE_LOCAL_LIST)

            # spearman_val = lmeasure.spearmans_rho()
            trust_val = lmeasure.trustworthiness()
            conti_val = lmeasure.continuity()
            mrre_xz_val = lmeasure.mrre_xz()
            mrre_zx_val = lmeasure.mrre_zx()

            values.extend(
                # [spearman_val, trust_val, conti_val, mrre_xz_val, mrre_zx_val]
                [trust_val, conti_val, mrre_xz_val, mrre_zx_val]
            )

        result = pd.DataFrame(
            {"measure": measures, "algorithm": algorithms, "values": values}
        )
        result = result.pivot(index="measure", columns="algorithm", values="values").fillna(
            "NA"
        )
        result.to_csv("./results/result.csv")
        print(f"{result}\n")
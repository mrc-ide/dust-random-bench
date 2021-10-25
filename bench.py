#!/usr/bin/env python3
import os
import subprocess

expected = [
    "engine",
    "distribution",
    "n_threads",
    "n_draws",
    "t_setup",
    "t_sample",
]


def run_sample(engine, distribution, n_threads, n_draws):
    exe = "./dustrand" if engine == "dust" else "./curand"
    res = subprocess.run(
        [exe, distribution, str(n_threads), str(n_draws)], capture_output=True
    )
    txt = res.stdout.decode("utf-8").strip().split(", ")
    key = [x.split(": ")[0] for x in txt]
    assert key == expected
    return ",".join([x.split(": ")[1] for x in txt]) + "\n"


def run_grid(engine, distribution, n_threads, n_draws):
    res = []
    for nt in n_threads:
        for nd in n_draws:
            print(f"{engine} {distribution} {nt} threads, {nd} draws")
            res.append(run_sample(engine, distribution, nt, nd))
    return res


if __name__ == "__main__":
    n_draws = [2 ** n for n in range(10, 22)]
    n_threads = [2 ** n for n in range(10, 22)]
    distribution = "uniform"
    res_curand = run_grid("curand", "uniform", n_threads, n_draws)
    res_dust = run_grid("dust", "uniform", n_threads, n_draws)

    with open("data/uniform.csv", "w") as f:
        header = ",".join(expected)
        f.write(header + "\n")
        f.writelines(res_curand)
        f.writelines(res_dust)

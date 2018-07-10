k=5
for s in range(100):
    for e in [5, 10, 20]:
        for l in [0.01, 0.001]:
            for m in ["auto", "auto_pos"]:
                CMD = "python script1.py -s {} -e {} -l {} -m {} -k {} -o {}".format(
                        s,
                        e,
                        l,
                        m,
                        k,
                        "results/{}-factor/{}/{}-{}".format(k, m, e, l))
                print(CMD)

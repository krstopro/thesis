for s in range(100):
    for e in [5, 10, 20]:
        for l in [0.01, 0.001]:
            for m in ["manual", "auto", "auto_pos"]:
                CMD = "python script1.py -s {} -e {} -l {} -m {} -o {}".format(
                        s,
                        e,
                        l,
                        m,
                        "results/{}/{}-{}".format(m, e, l))
                print(CMD)

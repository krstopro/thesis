k=1
g = 0.1
e=10
        
for s in range(10):
    #for g in [0.1]: #[0.1, 0.5, 1, 2]:
        #for e in [5, 10, 20]:
    for n in [1,2]:
        for d in [20,50,100]:
            for l in [0.01]:
                for m in ['lstm', 'gru', 'attn_lstm', 'attn_gru']:
                    output="results/{}-factor/{}/{}-{}-{}-{}".format(k, m, n, d, e, l)
                    CMD = "python script2.py -s {} -e {} -l {} -m {} -k {} -n {} -d {} -o {}".format(
                            s,
                            e,
                            l,
                            m,
                            k,
                            n,
                            d,
                            output)
                    print(CMD)

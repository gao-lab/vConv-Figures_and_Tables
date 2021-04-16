import Bio.SeqIO





if __name__ == '__main__':
    import glob
    import numpy as np

    CTCFfiles = glob.glob("../../../data/chip-seqFa/" + "**")

    fastasizelist = []
    outputName = []
    f = open("./fastause.txt","w")
    for filename in CTCFfiles:
        fsatasize = 0
        for record in Bio.SeqIO.parse(filename, "fasta"):
            fsatasize =fsatasize + len(str(record.seq))

        if fsatasize<400000:
            fastasizelist.append(fsatasize)
            outputName.append(filename.split("/")[-1]+"\n")

    arglist = np.argsort(np.asarray(fastasizelist))

    for i in range(10):
        f.write(outputName[arglist[i]])




import os
from tqdm import tqdm
BASE = "/mnt/zoltan/public/dataset/rawdata/val/"
NEW = "/mnt/zoltan/public/dataset/rawdata-mini/val/"
def main():
    nids = os.listdir( BASE )
    nids.sort()
    for nid in tqdm(nids):
        URL = BASE + nid + "/"
        cmd = "mkdir {}{}".format(NEW, nid)
        # print(cmd)
        os.system("rm -rf {}{}".format(NEW, nid))
        os.system(cmd)
        pics = os.listdir(URL)
        pics_to_copy = pics[:8]
        for p in pics_to_copy:
            cmd = "cp {}{}/{} {}{}/{}".format(BASE, nid, p, NEW, nid, p)
            os.system(cmd)
        
    return 

if __name__ == "__main__":
    main()
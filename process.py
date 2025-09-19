import torchvision

def frame_num(path):
    fw = open(path+'.csv', 'w', encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = line.strip()
            fn = item + '_mouth.mp4'
            vid, aud, infos = torchvision.io.read_video(fn, output_format='TCHW', pts_unit='sec')
            L = len(vid)
            fw.write(item+' '+str(L)+'\n')
    fw.close()


#frame_num('fulltrain.txt')
frame_num('test.txt')

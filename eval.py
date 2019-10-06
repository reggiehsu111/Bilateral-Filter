import numpy as np
import cv2
import argparse
import time

from joint_bilateral_filter import Joint_bilateral_filter

def main():
    parser = argparse.ArgumentParser(description='JBF evaluation')
    parser.add_argument('--sigma_s', default=3, type=int, help='sigma of spatial kernel')
    parser.add_argument('--sigma_r', default=0.1, type=float, help='sigma of range kernel')
    parser.add_argument('--input_path', default='./testdata/ex.png', help='path of input image')
    parser.add_argument('--gt_bf_path', default='./testdata/ex_gt_bf.png', help='path of gt bf image')
    parser.add_argument('--gt_jbf_path', default='./testdata/ex_gt_jbf.png', help='path of gt jbf image')
    
    args = parser.parse_args()

    img = cv2.imread(args.input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("img_rgb[0,0]",img_rgb[0,0])
    print("guidance[0,0]", guidance[0,0])
    # create JBF class
    JBF = Joint_bilateral_filter(args.sigma_s, args.sigma_r, border_type='reflect')

    # bilateral filter
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    # joint bilateral filter
    start_time = time.time()
    jbf_out = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)
    print("passed time:", time.time()-start_time)

    bf_gt = cv2.cvtColor(cv2.imread(args.gt_bf_path), cv2.COLOR_BGR2RGB)
    jbf_gt = cv2.cvtColor(cv2.imread(args.gt_jbf_path), cv2.COLOR_BGR2RGB)
    # print(img_rgb[0,:5,:])
    # print(jbf_out[0,:5,:])
    # print(jbf_gt[0,:5,:])

    jbf_output = cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.jpg', jbf_output)

    bf_error = np.sum(np.abs(bf_out-bf_gt))
    jbf_error = np.sum(np.abs(jbf_out-jbf_gt))
    
    print('%d %d'%(bf_error, jbf_error))
    print("diff:", np.sum(np.abs(jbf_out - bf_out)))
    # print(jbf_error)

    

if __name__ == '__main__':
    main()

from joint_bilateral_filter import *
import numpy as np
import cv2
import time
import pickle
from tqdm import tqdm
import threading
import sys
import os

class Optimizer:
    def __init__(self):
        self.weights = self.generate_weights()
        self.sigma_s = [1, 2, 3]
        self.sigma_r = [0.05, 0.1, 0.2]
        self.jbf = Joint_bilateral_filter(self.sigma_s[0], self.sigma_r[0])
        self.results = {}

    def generate_weights(self):
        ret = []
        for x in range(11):
            for y in range(11-x):
                z = 10-x-y
                ret.append(np.array([0.1*x,0.1*y,0.1*z]))
        return np.around(ret,decimals=2)

    def rgb2gray(self, image, weight):
        ret_img = np.matmul(image,weight).astype(np.uint8)
        return ret_img

    def _find_local_opt(self, w, ss, sr, input_img, bf_img, workers_i, store_dir):
        print(workers_i, "Starting thread with weights:", w)
        try:
            gray_img = self.rgb2gray(input_img, w)
            jbf = Joint_bilateral_filter(ss, sr)
            jbf_img = jbf.joint_bilateral_filter(input_img, gray_img)
            # cv2.imwrite('outputs/'+store_dir+'/'+str(ss)+'_'+str(sr)+'_'+'_'.join([str(x) for x in w])+'jbf_img.jpg', jbf_img)
            cost = np.sum(np.abs(jbf_img - bf_img))
            self.lock.acquire()
            print("sigma_s:", ss, "sigma_r:", sr, "weights:", np.around(w,decimals=2), "cost:", cost)
            try:
                self.results[str(ss)+'_'+str(sr)].append((w, cost))
            except KeyError:
                self.results[str(ss)+'_'+str(sr)] = [(w, cost)]

            self.lock.release()
        except KeyboardInterrupt:
            sys.exit()

    def find_local_optimal(self, input_img_path, store_dir):
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        for ss in tqdm(self.sigma_s):
            for sr in tqdm(self.sigma_r):
                self.jbf.set_sigma_s(ss)
                self.jbf.set_sigma_r(sr)
                bf_img = self.jbf.joint_bilateral_filter(input_img, input_img)
                # cv2.imwrite('outputs/'+store_dir+'/'+str(ss)+'_'+str(sr)+'bf_img.jpg', bf_img)
                workers = []
                self.lock = threading.Lock()
                workers_i = 0
                try:
                    for w in self.weights:
                        workers.append(threading.Thread(target = self._find_local_opt, args = (w, ss, sr,  input_img, bf_img, workers_i, store_dir)))
                        workers[workers_i].start()
                        workers_i += 1
                    for i in range(len(workers)):
                        workers[i].join()
                except (KeyboardInterrupt, SystemExit):
                    print("Exit!")
                    sys.exit()
                print("Done:", ss, sr)
        with open("outputs/all_results_"+store_dir+".pkl", 'wb') as file:
            pickle.dump(self.results, file)
        # with open("outputs/best_results_"+store_dir+".pkl", 'wb') as file:
        #     pickle.dump(self.local_bests, file)
        print("Processing time for", input_img_path, "is:", time.time()-start_time)

    def vote(self, output_dir, img_name):
        with open('outputs/all_results_'+img_name+'.pkl', 'rb') as file:
            all_results = pickle.load(file)
        # with open('outputs/best_results_'+img_name+'.pkl', 'rb') as file:
        #     best_results = pickle.load(file)
        # print("best results")
        # for key in best_results:
        #     print(key, best_results[key][0], best_results[key][1])
        all_local_mins = {}
        votes = {}
        neighbors = np.array([
            [0.1, -0.1, 0],
            [0.1, 0, -0.1],
            [0, 0.1, -0.1],
            [-0.1, 0.1, 0],
            [0, -0.1, 0.1],
            [-0.1, 0, 0.1]
            ])
        for key in all_results:
            # Dict to map weights to error
            weight_map = {}
            local_mins = []
            for pt in all_results[key]:
                weight_map['_'.join([str(x) for x in pt[0]])] = pt[1]
            # Find local minima
            for pt in all_results[key]:
                temp_weigths = pt[0]
                temp_neighbors = []
                for x in range(6):
                    temp = neighbors[x] + temp_weigths
                    if np.all(temp<=1) and np.all(temp>=0):
                        temp_neighbors.append(temp)
                min_flag = True
                for tn in temp_neighbors:
                    if weight_map['_'.join([str(x) for x in np.around(tn, decimals=2)])] < pt[1]:
                        min_flag = False
                if min_flag == True:
                    local_mins.append(pt)
            print("key:", key)
            print("local mins:", local_mins)
            all_local_mins[key] = local_mins
        for key in all_local_mins:
            for candidate in all_local_mins[key]:
                try:
                    votes['_'.join([str(x) for x in np.around(candidate[0], decimals=2)])] += 1
                except KeyError:
                    votes['_'.join([str(x) for x in np.around(candidate[0], decimals=2)])] = 1
        for key in votes:
            print(key, votes[key], )
        sorted_votes = sorted(votes.items(), key=lambda x:x[1], reverse=True)
        print(sorted_votes[:3])
        img_path = 'testdata/'+img_name+'.png'
        input_img = cv2.imread(img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        i = 0
        for x in sorted_votes[:3]:
            i+=1
            # Convert x to array
            weights = [float(i) for i in x[0].split('_')]
            gray_img = self.rgb2gray(input_img, weights)
            cv2.imwrite('outputs/bests/'+img_name+'/'+'best'+str(i)+'_'+x[0]+'.jpg', gray_img)
        gray_img = self.rgb2gray(input_img,(0.1,0.9,0))
        cv2.imwrite('outputs/bests/'+img_name+'/'+'best'+str(4)+'_0.1_0.9_0.0.jpg', gray_img)

    def check_find_local_optimal(self, input_img_path, store_dir):
        img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        ss = 2
        sr = 0.2
        self.jbf.set_sigma_s(ss)
        self.jbf.set_sigma_r(sr)
        print("range", self.jbf.range)
        print("window size", self.jbf.w_size)
        bf_img = self.jbf.joint_bilateral_filter(input_img, input_img)

        try:
            for w in self.weights:
                gray_img = self.rgb2gray(input_img, w)
                jbf_img = self.jbf.joint_bilateral_filter(input_img, gray_img)
                cost = np.sum(abs(jbf_img - bf_img))

                print("sigma_s:", ss, "sigma_r:", sr, "weights:", np.around(w,decimals=2), "cost:", cost)
                try:
                    self.results[str(ss)+'_'+str(sr)].append((w, cost))
                except KeyError:
                    self.results[str(ss)+'_'+str(sr)] = [(w, cost)]
        except (KeyboardInterrupt, SystemExit):
            print("Exit!")
            sys.exit()
        print("Done:", ss, sr)
        with open("outputs/all_results_check_"+str(ss)+'_'+str(sr)+store_dir+".pkl", 'wb') as file:
            pickle.dump(self.results, file)

        print("Processing time for", input_img_path, "is:", time.time()-start_time)


            # cv2.imwrite('outputs/bests/'+key+'.jpg', self.local_bests[key][2])


if __name__ == '__main__':
    opt = Optimizer()
    test_img = '1c'
    output_dir = 'outputs/bests/'+test_img
    # Output bf image
    ##########
    # input_img = cv2.imread('testdata/'+test_img+'.png')
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # jbf = Joint_bilateral_filter(1, 0.05)
    # bf_img = jbf.joint_bilateral_filter(input_img, input_img).astype(np.uint8)
    # bf_img = cv2.cvtColor(bf_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('outputs/bests/'+test_img+'/1_0.05_bf_img.jpg', bf_img)
    ##########
    try:
        os.mkdir(output_dir)
        os.mkdir('outputs/'+test_img)
    except:
        print("Directory can't be created")
        pass

    # opt.find_local_optimal('testdata/'+test_img+'.png', test_img)
    opt.vote(output_dir, test_img)
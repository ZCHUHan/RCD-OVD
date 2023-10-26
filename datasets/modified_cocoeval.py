import numpy as np
import datetime
import time
from collections import defaultdict
import pycocotools.mask as maskUtils
import copy
import json


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        print('--This is the modified cocoeval--')
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params

        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']

        ##这种声明方式产生的self._gts是一个字典，每个元素是列表
        ##这样得到的就是相同的img_id和类别id的信息，存放在一个列表中，即一张图像的同一个类别的框在一个列表中；
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        ## gts中一个gt格式：{"area": 735345, "iscrowd": 0, "image_id": 20190000781, "bbox": [225, 1052, 1257, 585], "category_id": 1, "id": 1063, "ignore": 0, "segmentation": []}
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        ## dts中一个dt格式：{"image_id": 20190000781, "category_id": 1, "bbox": [1584.88, 884.44, 152.43, 308.34], "score": 0.0, "segmentation": [[1584.88, 884.44, 1584.88, 1192.78, 1737.3100000000002, 1192.78, 1737.3100000000002, 884.44]], "area": 47000.2662, "id": 78100, "iscrowd": 0}
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

#        self.gt_id2img_id = {}
#        for gt_i in gts:
#            self.gt_id2img_id[gt_i['id']] = gt_i['image_id']

#        self.gt_imgid_cat_id = {}
#        for gt_i in gts:
#            if gt_i['image_id'] not in self.gt_imgid_cat_id.keys():
#                self.gt_imgid_cat_id[gt_i['image_id']] = {}
#                for cat in self.params.catIds:
#                    self.gt_imgid_cat_id[gt_i['image_id']][cat] = []
#            self.gt_imgid_cat_id[gt_i['image_id']][gt_i['category_id']].append(gt_i['id'])

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        # print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))  ##唯一imgid
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))  ##唯一gt类别id，不包括背景
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks

        ##self.ious是一个字典，每一个元素是表示一张图中某一个类别的预测框(m个)和这个类别的gt(n个)的iou矩阵(m,n)
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                     for imgId in p.imgIds
                     for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        ##self.evalImgs是列表，每一个元素是字典，存储的是单张图片，一种类别，特定areaRng下的预测框和gt的匹配结果(在不同的阈值下)
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        ##inds是score从大到小排列的索引
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        ##将此处的dt(一张图片一个类别的所有100个检测框(dt大于100个检测框的，按置信度取前100个(100个由p.maxDets设定))按置信度从大到小排列)
        ##注意是一张图片一种类别的预测框不超过p.maxDets[-1]个，而不是一张图片的预测框不超过这么多，除非设置忽视类别，那就等价于一张图片的总的预测框不多于p.maxDets[-1]
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')
        ##gt和dt是一张图片的一种类别的所有框信息；其中dt中只取p.maxDets[-1]个检测框，按置信度从大到小排序；
        ##g和d是从gt和dt中获取的segmentation信息(分割任务)，检测任务取得是bbox信息；
        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)  ##ious是(m,n),m是d的个数，即模型的预测检测框个数，n是g的框个数
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3];
            yg = g[1::3];
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2];
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3];
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3];
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            # 如果不满足该面积阈值就忽略
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]

        iscrowd = [int(o['iscrowd']) for o in gt]

        # load computed ious
        ## 满足阈值的gt与所有detection的iou, M*n(gtind)
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))  ##存储的是每一个iou阈值、p.maxDet[-1]下的gt能够匹配到的最大iou对应的模型预测框的id，匹配不到的值是0；
        dtm = np.zeros((T, D))  ##存储的是每一个iou阈值下的模型预测框匹配到的gt的id，匹配不到的是0；
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))  ##表示每一个阈值下的预测框匹配到的gt是否需要ignore

        ##dt已经按照置信度排过序，gt已经按照ignore排过位置，非ignore在前，ignore在后面
        ##下面的if里面实现的功能是每一个iou阈值下，遍历预测框(预测框已经按置信度从大到小排序)，一个预测框和gt匹配上，则
        ##另一个预测框不能再通过iou和这个gt进行匹配
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    # # 如果m= -1 代表这个dt没有得到匹配 m代表dt匹配的最好的gt的索引下标
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]  ##对应的能匹配上gt的预测框是否ignore
                    dtm[tind, dind] = gt[m]['id']  ##dt匹配上的gt的id
                    gtm[tind, m] = d['id']  ##gt中的框匹配上的预测框的id
        # set unmatched detections outside of area range to ignore
        ##将dtm中没有匹配到gt的预测框，同时预测框的area在指定的aRng范围外，则设置对应的预测框为ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,  ##aRng范围外的gt和未匹配到gt的预测框但在aRng范围外都是ignore，匹配到gt的预测框在aRng范围外正常计算，不ignore
            'maxDet': maxDet,  ##这里是p.maxDets[-1]
            'dtIds': [d['id'] for d in dt],  ##已经排过序的预测框id
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,  ##(T,D) 其中D是已经按置信度排除的bbox
            'gtMatches': gtm,  ##(T,G) G是按照aRng等信息排序的不ignore在前，ignore在后的gt
            'dtScores': [d['score'] for d in dt],  ##已经排过序的score
            'gtIgnore': gtIg,  ##G指的是单张图片特定aRng的gt是否ignore信息
            'dtIgnore': dtIg,  ##(T,D)
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)  ##设置的iou阈值的个数
        R = len(p.recThrs)  ##设置的召回的recThrs阈值的个数
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        G_num = 33152  ##设置的该评估用的数据集的gt总数,可以事先通过cvat查看标注的bbox个数，或者自己评估性的设置一个数
        precision = -np.ones(
            (T, R, K, A, M))  # -1 for the precision of absent categories ##这个是存储不同的rec值下的p值，相当于存储了pr曲线的采样点
        recall = -np.ones((T, K, A, M))
        precision_s = -np.ones((T, K, A, M))  ##真实的精确率值
        scores = -np.ones((T, R, K, A, M))
        DTMatch = -np.ones((T, K, A, G_num, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        ##根据self.evalImgs的存储形式，遍历时最里层是img_id、次外层是aRng、最外层是类别id
        '''
            self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                for catId in catIds
                for areaRng in p.areaRng
                for imgId in p.imgIds
            ]
        '''
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):  ##类别的索引下标遍历
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):  ##aRng的遍历
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    ##特定类别、特定aRng的所有图片中每一张图片的maxDet个预测框
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    ##是将特定类别，特定aRng的所有图片的预测框
                    # (每张图片特定类别、aRng取置信度从大到小的maxDet个框)
                    # 的置信度拉成一位数组，然后再次从大到小排列；
                    dtScoresSorted = dtScores[inds]
                    ##dtm、dtIg维度是(T,maxDet个数*图片个数)
                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])  ##gtIg维度是(图片个数，G)
                    npig = np.count_nonzero(gtIg == 0)  ##gt不ignore的个数
                    if npig == 0:
                        continue

                    ##dtm、dtIg维度是(T,maxDet个数*图片个数)
                    # 如果dtm对应的匹配gt不为0，且对应的gt没有被忽略，这个dt就是TP tips:[1,0,1,0,1,0]
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    # dtm对应的gt为0， 并且这个dt也没有被忽略，这个dt就是FP  tips:[0,1,0,1,0,1]
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    ###GT是特定类别、特定aRng、maxDet下所有图片的gt能被预测到的情况
                    # GT = [self.evalImgs[Nk + Na + i] for i in i_list]
                    gtmatch_id = tps * dtm  ##(T,图片个数*maxDet)
                    indice_gt = [np.where(i > 0) for i in gtmatch_id]
                    unique_id = np.array([np.unique(i[indice_gt[p]]) for p, i in
                                          enumerate(gtmatch_id)])  # (T,d（不同的iou下，maxDet下的预测框能检测到的gt，所以维度d维度不一样）)
                    # gtmatch = np.concatenate([e['gtMatches'] for e in GT], axis=1)  ##(T,图片个数*G)
                    # gt_total_num_k_a = gtmatch.shape[1]
                    # (T,K,A,G_num,M)
                    for j, id in enumerate(unique_id):
                        gt_total_num_k_a = len(id)
                        DTMatch[j, k, a, :gt_total_num_k_a, m] = id  ##存储的是预测框能匹配上的gt的id

                    #按照行的方式(每个IOU阈值下)进行匹配到的累加, 每个INDEX就是这个置信度下匹配到的tp, fp
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))  ##特定召回率下的precision值(pr曲线)
                        ss = np.zeros((R,))  ##特定召回率下的对应的bbox的置信度

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                            precision_s[t, k, a, m] = pr[-1]
                        else:
                            recall[t, k, a, m] = 0
                            precision_s[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist();
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        ##这里调用的np.searchsorted表示p.recThrs中每一个值能插入到rc中的位置索引，其中rc必须是升序
                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,  ##(T,R,K,A,M)
            'recall': recall,  ##(T,K,A,M)
            'precision_s': precision_s,
            'scores': scores,  ##(T,R,K,A,M)
            'DTMatch': DTMatch,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def get_good_predict_data(self, in_gt_imgid_cat_id, gt_id2img_id):
        '''
        该函数主要用于得到评估数据集中gt成功预测的图片，相反可以得到gt预测不好的图片用于离线困难数据挑选；
        '''

        def get_imgid_excellent_predict(save_path, iouThr, areaRng, maxDets, catId):

            p = self.params
            ##(T,K,A,G_num,M)
            DTMatch = self.eval['DTMatch'] ##存储的是预测框能匹配上的gt的id
            # print('DTMatch', DTMatch)
            # print(DTMatch.shape) # (10, 65, 4, 33152, 3)
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            cind = [i for i, cat in enumerate(p.catIds) if cat in catId]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                DTMatch = DTMatch[t]
            #print(DTMatch.shape)
            ##应该分类别计算，不然统一取unique的话，一张图中只要有一种类别的一个gt被检测出来，这张图片后续就会认为是预测较好
            ##的样本，但是该图片中同一种类的其他gt或者其他类的gt可能完全没检出，因此需要分类别计算
            ##iou阈值混在一起没问题，或者取特定的iou阈值就可以了
            ##其实也就是将gt的id和这里的np.unique(DTMatch[:,cind,aind,:,mind])取差集就知道漏检情况了，因为id是指的gt的框的索引
            gt_imgid_cat_id = copy.deepcopy(in_gt_imgid_cat_id)
            # print('gt_imgid_cat_id:', gt_imgid_cat_id)

            DTMatch = DTMatch[:, cind, aind, :, mind]
            # print(DTMatch.shape, DTMatch)
            for i, cat in enumerate(catId):
                print('i', i)
                DTMatch_catid = np.unique(DTMatch[i, :, :])
                DTMatch_catid = np.delete(DTMatch_catid, 0)
                # print('DTMatch_catid', DTMatch_catid)
                for j in DTMatch_catid:
                    print('DTMatch_catid', type(j), j)
                    #print('gt_id2img_id type', gt_id2img_id.keys())
                    image_id = gt_id2img_id[int(j)]
                    gt_imgid_cat_id[image_id][cat].remove(j)

            null_leak_det = []  ##存储完全检测出gt bbox的图片id
            for image_id_i in gt_imgid_cat_id.keys():
                num_empty = 0
                for catId_i in catId:
                    # for catId_i in gt_imgid_cat_id[image_id_i]:
                    if len(gt_imgid_cat_id[image_id_i][catId_i]) == 0:
                        num_empty += 1
                        # RuntimeError: dictionary changed size during iteration
                        # gt_imgid_cat_id[image_id_i].pop(catId_i)
                # if num_empty == len(self.params.catIds):
                if num_empty == len(catId):
                    null_leak_det.append(image_id_i)

            det_gt = np.array([self.cocoGt.loadImgs(ids=[i])[0]['file_name'] for i in null_leak_det])
            det_gt = np.unique(det_gt)
            np.savetxt(save_path, det_gt, fmt='%s', delimiter=',')

            # json_str = json.dumps(gt_imgid_cat_id)
            for img_id_i in null_leak_det:
                gt_imgid_cat_id.pop(img_id_i)

            # print('length of gt_imgid_cat_id={}, none_leak_det={}, imgIds={}'.format(len(gt_imgid_cat_id.keys()),len(null_leak_det),len(self.params.imgIds)))
            json_str = repr(gt_imgid_cat_id)
            with open(save_path.replace('.txt', '.json').replace('good', 'leak_det'), 'w') as json_file:
                json_file.write(json_str)

            leak_det_gt = gt_imgid_cat_id.keys()
            leak_det_gt = np.array([self.cocoGt.loadImgs(ids=[i])[0]['file_name'] for i in leak_det_gt])
            leak_det_gt = np.unique(leak_det_gt)
            np.savetxt(save_path.replace('good', 'leak_det'), leak_det_gt, fmt='%s', delimiter=',')
            # DTMatch = np.unique(DTMatch[:,cind,aind,:,mind])
            # DTMatch = np.delete(DTMatch,0)
            # ###检测框对应的gt id和真实的gt的id的差集就是未检测出的gt的框的id
            # #DTMatch.tolist().remove(-1.0)
            # #print(DTMatch)
            # det_gt = np.array([ self.cocoGt.loadImgs(ids=[gt_id2img_id[i]])[0]['file_name'] for i in DTMatch])
            # det_gt = np.unique(det_gt)

            # # with open(save_path,'w+') as file_object:
            # #     json.dump(DTMatch,file_object)
            # np.savetxt(save_path, det_gt, fmt='%s', delimiter=',')
            # # f = open(save_path,'w+')
            # # for i in DTMatch:
            # #     f.write(i)
            # #     f.write('\n')
            # # f.close()
            # print('save iou={}| areaRng={}| maxDets={}| catId={} to {}'.format(iouThr, areaRng, maxDets, catId, save_path))

        save_path = r'./output/good_predict.txt'
        get_imgid_excellent_predict(save_path, iouThr=.5, areaRng='all', maxDets=self.params.maxDets[0],
                                    catId=self.params.unseen_catIds)

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100, catId=self.params.catIds):
            '''
            precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories ##这个是存储不同的rec值下的p值，相当于存储了pr曲线的采样点
            recall      = -np.ones((T,K,A,M))
            precision_s = -np.ones((T,K,A,M))   ##真实的精确率值
            '''
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            # titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            # typeStr = '(AP)' if ap==1 else '(AR)'
            # titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            # typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            cind = [i for i, cat in enumerate(p.catIds) if cat in catId]

            if ap == 1:
                titleStr = 'Average P-R curve Area'
                typeStr = '(mAP)'
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # s = s[:,:,:,aind,mind]
                s = s[:, :, cind, aind, mind]
            elif ap == 0:
                titleStr = 'Average Recall'
                typeStr = '(AR)'
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # s = s[:,:,aind,mind]
                s = s[:, cind, aind, mind]

            else:
                titleStr = 'Average Precision'
                typeStr = '(AP)'
                # dimension of precision: [TxKxAxM]
                s = self.eval['precision_s']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # s = s[:,:,aind,mind]
                s = s[:, cind, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            '''
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.902
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.985
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.975
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.902
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.687
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.932
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.932
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.932

            '''

            '''
            Average P-R curve Area (mAP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.902
            Average P-R curve Area (mAP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.985
            Average P-R curve Area (mAP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.975
            Average P-R curve Area (mAP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
            Average P-R curve Area (mAP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
            Average P-R curve Area (mAP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.902
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.687
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.932
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.932
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.932
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.936
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.127
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.013
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.988
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.136
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=  1 ] = 0.981
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 10 ] = 0.135

            '''
            # stats = np.zeros((12,))
            stats = np.zeros((77,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])

            # base or novel
            stats[6] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2], catId=self.params.seen_catIds)
            stats[7] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2], catId=self.params.unseen_catIds)
            stats[8] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2], catId=self.params.seen_catIds + self.params.unseen_catIds)

            # AR
            # stats[8] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[0])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2], catId=self.params.unseen_catIds)
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2], catId=self.params.unseen_catIds)
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2], catId=self.params.unseen_catIds)

            # each cat
            stats[12] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[0]])
            stats[13] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[1]])
            stats[14] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[2]])
            stats[15] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[3]])
            stats[16] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[4]])
            stats[17] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[5]])
            stats[18] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[6]])
            stats[19] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[7]])
            stats[20] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[8]])
            stats[21] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[9]])
            stats[22] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[10]])
            stats[23] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[11]])
            stats[24] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[12]])
            stats[25] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[13]])
            stats[26] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[14]])
            stats[27] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[15]])
            stats[28] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[16]])

            stats[29] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[17]])
            stats[30] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[18]])
            stats[31] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[19]])
            stats[32] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[20]])
            stats[33] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[21]])
            stats[34] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[22]])
            stats[35] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[23]])
            stats[36] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[24]])
            stats[37] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[25]])
            stats[38] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[26]])
            stats[39] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[27]])
            stats[40] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[28]])
            stats[41] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[29]])
            stats[42] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[30]])
            stats[43] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[31]])
            stats[44] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[32]])
            stats[45] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[33]])
            stats[46] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[34]])
            stats[47] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[35]])
            stats[48] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[36]])
            stats[49] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[37]])
            stats[50] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[38]])
            stats[51] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[39]])
            stats[52] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[40]])
            stats[53] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[41]])
            stats[54] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[42]])
            stats[55] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[43]])
            stats[56] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[44]])
            stats[57] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[45]])
            stats[58] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[46]])
            stats[59] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.seen_catIds[47]])

            stats[60] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[0]])
            stats[61] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[1]])
            stats[62] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[2]])
            stats[63] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[3]])
            stats[64] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[4]])
            stats[65] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[5]])
            stats[66] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[6]])
            stats[67] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[7]])
            stats[68] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[8]])
            stats[69] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[9]])
            stats[70] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[10]])
            stats[71] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[11]])
            stats[72] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[12]])
            stats[73] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[13]])
            stats[74] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[14]])
            stats[75] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[15]])
            stats[76] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=[self.params.unseen_catIds[16]])


            # stats = np.zeros((2,))
            # stats[0] = _summarize(0, iouThr=.8, maxDets=self.params.maxDets[0])
            # stats[1] = _summarize(2, iouThr=.8, maxDets=self.params.maxDets[0])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    '''
    Params for coco evaluation api
    '''

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        # array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

        # for open-vocabulary setting
        self.unseen_catIds = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]
        self.seen_catIds = [1, 2, 3, 4, 7, 8, 9, 15, 16, 19, 20, 23, 24, 25, 27, 31, 33, 34, 35, 38, 42, 44, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 65, 70, 72, 73, 74, 75, 78, 79, 80, 82, 84, 85, 86, 90]

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

# unseen ids [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]
# seen ids [1, 2, 3, 4, 7, 8, 9, 15, 16, 19, 20, 23, 24, 25, 27, 31, 33, 34, 35, 38, 42, 44, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 65, 70, 72, 73, 74, 75, 78, 79, 80, 82, 84, 85, 86, 90]
# all ids [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]
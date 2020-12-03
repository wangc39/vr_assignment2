import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
import copy

def getColor():
    color: int
    color1 = ri(16, 255)
    color2 = ri(16, 255)
    color3 = ri(16, 255)
    color1 = hex(color1)
    color2 = hex(color2)
    color3 = hex(color3)
    ans = "#" + color1[2:] + color2[2:] + color3[2:]
    return ans

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

class myVoronoi:

    def __init__(self, x=None, y=None):
        # np.random.seed(0)
        # self.N = 4
        # self.dots = np.random.randn(self.N, 2)

        self.lower = 0
        self.upper = 50
        self.dots = np.concatenate([np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)], axis=1)
        self.N = self.dots.shape[0]

        self.based_path = os.path.abspath(os.path.dirname(__file__)) # 获取代码运行的基本路径

        # 如果存储图片的文件夹不存在的话，就创建该文件夹
        if not os.path.exists(os.path.join(self.based_path, "./figures")):
            os.makedirs(os.path.join(self.based_path, "./figures"))

        triangleMat, edgeMat = self.createDelaulay()
        
        self.plot_figure(triangleMat)

        self.createVoronoi(triangleMat)

    def createVoronoi(self, triangleMat):

        triangleCenterMat = self._getTriangleCenterMat(triangleMat)
        
        borderTriangleMat, borderPoint, triangleTempMat = self._makeBorderTriangle(triangleMat)
        # print("borderPoint", borderPoint)
        # print("*"*30)

        # 绘制散点图
        plt.figure(num = 2, figsize=(8, 8), facecolor='w')

        # 进行边缘检测
        for i in range(3, self.dots.shape[0]):
            tempTriangle, consTempTriangle, mask = self._findPointTriangle(i, triangleMat)
            # print("consTempTriangle:", consTempTriangle, type(consTempTriangle))


            plt.scatter(triangleCenterMat[consTempTriangle, 0], triangleCenterMat[consTempTriangle, 1], color="r", s=20)

            if mask:
                plt.fill(triangleCenterMat[consTempTriangle, 0], triangleCenterMat[consTempTriangle, 1], color=randomcolor())
            # 需要做边缘化处理
            else:
                tempTriangle, consTempTriangle, delLeft, delRight = self._selectTempTriangle(tempTriangle, consTempTriangle, triangleCenterMat)
                # print(delLeft)
                flag = True
                # 获取边缘线段
                tempBorderEdge = triangleCenterMat[consTempTriangle, :]
                # 边缘延长线1
                if delLeft == 0:
                    tempEdge = tempTriangle[0, [0, 1]]
                    tempBorderDot1 = self._edgePointFind(tempEdge, triangleCenterMat[consTempTriangle[0], :], borderPoint)
                    if tempBorderDot1.shape[0] == 0: flag = False # 由于边缘点超出就不再画出
                else:
                    tempBorderDot1 = self._edgePointFindrd(triangleCenterMat[consTempTriangle[0], :], triangleCenterMat[delLeft, :])
                tempBorderEdge = np.append(tempBorderDot1.reshape(1, -1), tempBorderEdge, axis=0) # 按照行连接

                # 边缘延长线2
                if delRight == 0:
                    tempEdge = tempTriangle[-1, [0, 2]]
                    tempBorderDot2 = self._edgePointFind(tempEdge, triangleCenterMat[consTempTriangle[-1], :], borderPoint)
                    if tempBorderDot2.shape[0] == 0: flag = False # 边缘点超出不在画出
                else:
                    tempBorderDot2 = self._edgePointFindrd(triangleCenterMat[consTempTriangle[-1], :], triangleCenterMat[delRight, :])
                tempBorderEdge = np.append(tempBorderEdge, tempBorderDot2.reshape(1, -1), axis=0) # 按照行连接
                
                # 绘制边缘图形
                if flag:
                    if tempBorderDot1[0] != tempBorderDot2[0] and tempBorderDot1[1] != tempBorderDot2[1]:
                        # 求交点三
                        tempBorderEdge = np.append(tempBorderEdge, self._makeTempBorder(tempBorderDot1, tempBorderDot2).reshape(1, -1), axis=0)

                    # 绘制图形
                    plt.fill(tempBorderEdge[:, 0], tempBorderEdge[:, 1], color=randomcolor())


        plt.xlim([0, self.upper])
        plt.ylim([0, self.upper])
        plt.title("Voronoi")
        plt.show()


    def _makeTempBorder(self, tempBorderPoint1, tempBorderPoint2):
        tempBorderPoint3 = np.zeros(shape=[2])

        # 检测边缘点
        for i in range(len(tempBorderPoint1)):
            if tempBorderPoint1[i] == 0 or tempBorderPoint1[i] == self.upper:
                tempBorderPoint3[i] =  tempBorderPoint1[i]
            
            # 对第二个点也是相同的道理
            if tempBorderPoint2[i] == 0 or tempBorderPoint2[i] == self.upper:
                tempBorderPoint3[i] =  tempBorderPoint2[i]

        return tempBorderPoint3

        

    def _edgePointFindrd(self, xy1, xy2):
        xy3, xy4 = self._selectEdge(xy1, xy2)
        tempBorderDot1 = self._cross_point(xy1, xy2, xy3, xy4)

        return tempBorderDot1


    def _edgePointFind(self, tempEdge, xy, borderPoint):
        # 做出一点对边缘的中垂线，得到边缘点坐标

        x = xy[0]
        y = xy[1]
        # 判断中心点是否在大图形中
        dpoint = np.array([borderPoint[0, 0]])
        dpoint = np.hstack((dpoint, borderPoint[:, 1]))

        # dpoint.append(list(borderPoint[:, 1]))
        mask = self._pointInPoly(dpoint, xy)

        # 求边的中心点
        xz = (self.dots[tempEdge[0], 0] + self.dots[tempEdge[1], 0]) / 2
        yz = (self.dots[tempEdge[0], 1] + self.dots[tempEdge[1], 1]) / 2

        flag = 1
        if mask:
            # 如果点在内部 做中心到边缘延长线即可
            xy1 = copy.deepcopy(xy)
            xy1 += [ i * 2 / math.sqrt((xz - x) ** 2 + (yz - y) ** 2) for i in [xz - x, yz - y] ]
        else:
            # 点在外部 但是没有超过边界
            if not ((x < 0) or (x > self.upper) or (y < 0) or (y > self.upper)):
                xy1 = copy.deepcopy(xy)
                xy1 -= [ i * 2 / math.sqrt((xz - x) ** 2 + (yz - y) ** 2) for i in [xz - x, yz - y] ]
            else:
                # 在外界
                flag = False
                tempBorderdot = np.empty(shape=(1, 1), dtype=np.int)
        
        if flag:
            # 判断4个边是哪个
            xy2, xy3 = self._selectEdge(xy, xy1)
            # 求两点求交运算
            tempBorderdot = self._cross_point(xy, xy1, xy2, xy3)

        return tempBorderdot


    def _cross_point(self, xy, xy1, xy2, xy3):
        # 计算两点之间的交点
        x1, y1 = xy[0], xy[1]
        x2, y2 = xy1[0], xy1[1]
        x3, y3 = xy2[0], xy2[1]
        x4, y4 = xy3[0], xy3[1]
        
        k1=(y2-y1)*1.0/(x2-x1)#计算k1,由于点均为整数，需要进行浮点数转化
        b1=y1*1.0-x1*k1*1.0#整型转浮点型是关键
        if (x4-x3)==0:#L2直线斜率不存在操作
            k2=None
            b2=0
        else:
            k2=(y4-y3)*1.0/(x4-x3)#斜率存在操作
            b2=y3*1.0-x3*k2*1.0
        if k2==None:
            x=x3
        else:
            x=(b2-b1)*1.0/(k1-k2)
        y=k1*x*1.0+b1*1.0
        return np.array([x,y])


    def _selectEdge(self, xy, xy1):
        # 判断射线会和那条边相交
        deg1 = np.angle(complex(0-xy[0], 0-xy[1]))
        deg2 = np.angle(complex(self.upper-xy[0], 0-xy[1]))
        deg3 = np.angle(complex(self.upper-xy[0], self.upper-xy[1]))
        deg4 = np.angle(complex(0-xy[0], self.upper-xy[1]))
        deg0 = np.angle(complex(xy1[0]-xy[0], xy1[1]-xy[1]))

        # 使用np.hstach将一维数组连接起来
        idx = np.argsort(np.hstack([deg0, deg1, deg2, deg3, deg4]))
        k = np.where(idx == 0)[0]



        if k == 0:
            xy2 = np.array([0, 0])
            xy3 = np.array([0, self.upper])
        elif k == 1:
            xy2 = np.array([0, 0])
            xy3 = np.array([self.upper, 0])
        elif k == 2:
            xy2 = np.array([self.upper, 0])
            xy3 = np.array([self.upper, self.upper])
        elif k == 3:
            xy2 = np.array([0, self.upper])
            xy3 = np.array([self.upper, self.upper])
        else:
            xy2 = np.array([0, 0])
            xy3 = np.array([0, self.upper])
        

        return xy2, xy3


    def _pointInPoly(self, dpoint, xy):
        Ndot = self.dots[dpoint, :]
        PN = Ndot - xy
        TN = np.zeros(shape=(len(Ndot), 1))
        for i in range(len(Ndot) - 1):
            TN[i] = self._crossDot(PN[i, :], PN[i+1, :])
        
        if np.abs(np.sum(np.sign(TN))) == len(Ndot) - 1:
            mask = 1
        else:
            mask = 0
        return mask

    def _crossDot(self, point1, point2):
        # 两个向量之间做差积
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        return (x1 * y2 - y1 * x2)

    def _selectTempTriangle(self, tempTriangle, consTempTriangle, triangleCenterMat):
        """
        删除所有中心点超出边界的三角形
        """

        delLeft = 0
        delRight = 0
        for i in range(tempTriangle.shape[0]):
            centerDots = triangleCenterMat[consTempTriangle[i], :]
            if centerDots[0] > 0 and centerDots[0] < self.upper and centerDots[1] < self.upper and centerDots[1] > 0:
                break
        
        if i != 0:
            delLeft = consTempTriangle[i-1]
            tempTriangle = np.delete(tempTriangle, list(range(i)), axis=0)
            consTempTriangle = np.delete(consTempTriangle, list(range(i)))
            # del consTempTriangle[list(range(i-1))]  # 删除对应的下标

        # 将矩阵倒着来一遍
        tempTriangle = np.flipud(tempTriangle)
        consTempTriangle = np.flipud(consTempTriangle)

        for j in range(tempTriangle.shape[0]):
            centerDots = triangleCenterMat[consTempTriangle[j], :]
            if centerDots[0] > 0 and centerDots[0] < self.upper and centerDots[1] < self.upper and centerDots[1] > 0:
                break

        if j != 0:
            delRight = consTempTriangle[j-1]
            tempTriangle = np.delete(tempTriangle, list(range(j)), axis=0)
            consTempTriangle = np.delete(consTempTriangle, list(range(j)))
        
        # 将三角形数组转置回来
        tempTriangle = np.flipud(tempTriangle)
        consTempTriangle = np.flipud(consTempTriangle)

        return tempTriangle, consTempTriangle, delLeft, delRight


    def _findPointTriangle(self, k, triangleMat):
        consTempTriangle = np.arange(triangleMat.shape[0])
        mask = True
        idx = self._findANumber(triangleMat, k)
        tempTriangleMat = triangleMat[idx, :]
        consTempTriangle = np.array(idx)

        # 将k放到每行第一个
        for m in range(tempTriangleMat.shape[0]):
            if tempTriangleMat[m][0] != k:
                tempIndex = list(np.where(tempTriangleMat[m] == k)[0])[0]
                tempTriangleMat[m, [0, tempIndex]] = tempTriangleMat[m, [tempIndex, 0]]
        # 如果有一个点只出现过一次，把这个点包含的三角形放到第一行
        nums = self._findNumsAppearOnce(tempTriangleMat.reshape(-1))

        if len(nums) != 0:
            num = np.sort(nums)[0]
            idx = self._findANumber(tempTriangleMat, num)[0] # 只能找到一个数字 所以直接取值
            tempTriangleMat[[0, idx], :] = tempTriangleMat[[idx, 0], :]
            consTempTriangle[[0, idx]] = consTempTriangle[[idx, 0]]
            mask = False
            if tempTriangleMat[0, 2] == num:
                tempTriangleMat[0, [1, 2]] = tempTriangleMat[0, [2, 1]]
        
        # 然后将首尾排序
        for i in range(1, tempTriangleMat.shape[0]):
            idx = self._findANumber(tempTriangleMat, tempTriangleMat[i-1, 2])
            idx.remove(idx[0])
            idx = idx[0]
            tempTriangleMat[[idx, i], :] = tempTriangleMat[[i, idx], :]
            if tempTriangleMat[i-1, 2] != tempTriangleMat[i, 1]:
                tempTriangleMat[i, [1, 2]] = tempTriangleMat[i, [2, 1]]
            consTempTriangle[[i, idx]] = consTempTriangle[[idx, i]]

        return tempTriangleMat, consTempTriangle, mask

    def _findNumsAppearOnce(self, array):
        # write code here
        dic = {}
        for i in array:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
        res = []
        for i in dic:
            if dic[i] == 1:
                res.append(i)
        return res

    def _findANumber(self, arr, k):
        """
        返回k在arr中的下标
        输入参数:
            arr：数组 注意是二维数组！！
        输出参数：
            idx 下标数组
        """
        idx = []
        # print(arr.shape[1])
        # print(arr)
        # print(arr[0])
        for i in range(arr.shape[0]):
            if k in list(arr[i]):
                idx.append(i)
        return idx

    def _getTriangleCenterMat(self, triangleMat):
        """
        获取三角形的外接圆的圆心位置
        输入参数:
            triangleMat: 三角形
        输出参数:
            triangleCenterMat: 每个三角形的外接圆的圆心位置
        """  
        triangleCenterMat = np.empty(shape=[0, 2], dtype=np.int) # 三角形外接圆的圆心位置
        for i in range(triangleMat.shape[0]):
            # 调用函数_getCircle生成三角形的外接圆 返回圆心的位置和半径
            a, b, _ = self._getCircle(self.dots[triangleMat[i, 0], :], self.dots[triangleMat[i, 1], :], 
                                                self.dots[triangleMat[i, 2], :])
            triangleCenterMat = np.append(triangleCenterMat, np.array([a, b]).reshape(1, -1), axis=0)
        
        return triangleCenterMat
        
    def _makeBorderTriangle(self, triangleMat):

        borderTriangleMat = np.empty(shape=[0, triangleMat.shape[1]], dtype=np.int)
        borderPoint = np.empty(shape=[0, 2], dtype=np.int)
        triangleTempMat = np.zeros(triangleMat.shape)

        for i in range(triangleMat.shape[0]):
            tempTriangle = triangleMat[i, :] # 获取临时三角形

            # 判断三角形的12 23 13点
            for j in range(triangleMat.shape[1]):
                borderPoint, triangleTempMat = self._generateBorderPoint(triangleMat, tempTriangle,
                                                                 borderPoint, triangleTempMat, i, j)

            if ~np.all(triangleTempMat[i, :]):
                # 如果边缘三角形少于三个就添加
                borderTriangleMat = np.append(borderTriangleMat, tempTriangle.reshape(1, -1), axis=0)    

        # 对得到的borderPoint进行排序输出
        for i in range(borderPoint.shape[0]-1):
            idx = self._returnIndex(borderPoint, i)
            borderPoint[[i+1, idx], :] = borderPoint[[idx, i+1], :] # 交换顺序
            if borderPoint[i, 1] == borderPoint[i+1, 1]:
                borderPoint[i+1, [0, 1]] = borderPoint[i+1, [1, 0]] # 交换下一点的顺序
        return borderTriangleMat, borderPoint, triangleTempMat

    def _returnIndex(self, borderPoint, k):
        idx = []
        for i in range(borderPoint.shape[0]):
            if k != i and borderPoint[k, 1] in borderPoint[i]:
                idx.append(i)
        if len(idx) == 0:
            return idx
        else:
            return idx[0]
        # return idx

    def _generateBorderPoint(self, triangleMat, triangle, borderPoint, triangleTempMat, m, k):

        """
        找到三角形中的找到临时边，也就是不和其他三角形连接的边
        输入参数：
            triangleMat: 总三角形的集合
            triange: 需要判断的三角形，也就是我们遍历的三角形
            borderPoint: 边界点
            triangleTempMat: 临时三角形的集合
            m: 遍历中的第几个三角形
            k: 需要判断的三角形的第k个点
        输出参数:
            borderPoint: 临界边上的点
            triangleTempMat
            调试使用
        """
        idx = []
        for i in range(triangleMat.shape[0]):
            if i != m and triangle[k] in triangleMat[i] and triangle[(k+1) % triangle.shape[0]] in triangleMat[i]:
                idx.append(i)
        if len(idx) == 0:
            # 如果这条边没有和别的三角形相连接 就说明这条边是临时边
            borderPoint = np.append(borderPoint, np.array([triangle[k], triangle[(k + 1) % triangle.shape[0]]]).reshape(1, -1), axis=0)
        elif len(idx) == 1:
            triangleTempMat[m, k] = idx[0]
        
        return borderPoint, triangleTempMat

    def createDelaulay(self):
        # 创建Delaulay三角形

        # 对二位数组进行自定义排序 按照行排序一列、二列依次排序
        # 这里千万不能使用np.sort 后悔一生
        self.dots = np.array(sorted(self.dots.tolist(), key=lambda x: (x[0], x[1])))
        # 找出最大包含的三角形
        # axis-0 表示按照列返回最大值或者最小值
        xmin, ymin = np.min(self.dots, axis=0)
        xmax, ymax = np.max(self.dots, axis=0)

        # print(xmin, xmax)
        
        # 创建最大的三角形，这个三角形正好能包住所有点
        maxTriangle = np.array([[(xmin+xmax)/2 -(xmax-xmin)*1.5, ymin-(xmax-xmin)*0.5], 
                            [(xmin+xmax)/2,ymax+(ymax-ymin)+(xmax-xmin)*0.5],  [(xmin+xmax)/2+(xmax-xmin)*1.5,ymin-(xmax-xmin)*0.5]])
        # print(maxTriangle)
        self.dots = np.concatenate([maxTriangle, self.dots], axis=0) # axis=0表示按照行进行拼接
        # print(self.dots)

        # 点的集合获得最大三角形的三个点
        edgeMat = np.array([self.__convertData(0, 1, self.dots[0, :], self.dots[1, :]), 
                            self.__convertData(1, 2, self.dots[1, :], self.dots[2, :]),
                            self.__convertData(0, 2, self.dots[0, :], self.dots[2, :])])
        # print(edgeMat)
        triangleMat = np.array([0, 1, 2]).reshape(1, -1) # 三角集合 包含三个点
        tempTriangleMat = np.array([0, 1, 2]).reshape(1, -1) # 临时三角形

        # 开始遍历 因为self.dots前三个点是最大三角形的三个点
        for i in range(3, self.N + 3):
            tempPoint = self.dots[i, :] # 初始化第一个点
            tempDel = [] # 初始化要删除的
            tempEdgeMat = np.empty(shape=[0, 6], dtype=np.int) # 初始化临时边
            # print("tempEdgeMat.shape: ", tempEdgeMat.shape)
            for j in range(tempTriangleMat.shape[0]):
                # print(tempTriangleMat.shape[0])
                # print(tempTriangleMat[j, 0])
                mask = self._locationPoint(self.dots[tempTriangleMat[j, 0], :], self.dots[tempTriangleMat[j, 1], :],
                                                 self.dots[tempTriangleMat[j, 2], :], tempPoint)
                # print("mask: ", mask)
                if mask == 2:
                    # 点在三角形外接圆的外部 说明该三角形就是Delaunay

                    # 将该三角形添加到正式的三角形中 axis=0表示行添加
                    # print("tempTriangleMat[j, :]", tempTriangleMat[j, :])
                    # print("triangleMat", triangleMat)
                    triangleMat = np.concatenate([triangleMat, tempTriangleMat[j, :].reshape(1, -1)], axis=0) 
                    # tempDel = np.concatenate([tempDel, j], axis=0)
                    tempDel.append(j)

                    # 将新的三角形的边添加到edgeMat中
                    edgeMat = np.concatenate([edgeMat, self._makeEdge(tempTriangleMat[j, 0], tempTriangleMat[j, 1], tempTriangleMat[j, 2])])
                    # print("edgeMat:\n", edgeMat)
                    # 对edgeMat进行不排序的去重 从而实现去除重复边
                    edgeMat = self._unranked_unique(edgeMat)
                elif mask == 0:
                    # 如果点在三角形外接圆的内部 说明该三角形不是Delaunay三角形

                    # 形成三个顶点和遍历的第i个点的三条线段
                    tempEdge = self._makeTempEdge(tempTriangleMat[j, 0], tempTriangleMat[j, 1], tempTriangleMat[j, 2], i)
                    # print("tempEdge: ", tempEdge)
                    # 将新加入的线段加入到tempEdgeMat中
                    tempEdgeMat = np.append(tempEdgeMat, tempEdge, axis=0)
                    # print("tempEdgeMat:", tempEdgeMat)
                    # print("***"*30)
                    # tempDel = np.concatenate([tempDel, j], axis=0)
                    tempDel.append(j)
                else:
                    continue
            
            # 移除加入的临时三角形
            tempTriangleMat = np.delete(tempTriangleMat, tempDel, axis=0) # 删除要去掉的行

            # 检查每一个行中是否为全0元素
            flags = np.any(tempTriangleMat, axis=1).tolist()
            idxArr = [i for i, x in enumerate(flags) if x == False]
            tempTriangleMat = np.delete(tempTriangleMat, idxArr, axis=0)
            
            tempTriangleMat = self._detectArray(tempTriangleMat, dim=3)
            # print("tempTriangleMat:", tempTriangleMat)
            # 对tempEdgeMat进行去重并且保持原先的顺序不变
            tempEdgeMat = self._unranked_unique(np.array(tempEdgeMat))
            # print("tempEdgeMat", tempEdgeMat)
            # print("tempEdgeMat.shape", tempEdgeMat.shape)
            # print("***"*30)
            tempTriangleMat = np.append(tempTriangleMat, self._makeTempTriangle(np.array(tempEdgeMat), i), axis=0)
            # print("tempTriangleMat", tempTriangleMat)

        # 遍历完成之后 合并三角形的集合
        triangleMat = np.concatenate([triangleMat, tempTriangleMat], axis=0)
        edgeMat = np.concatenate([edgeMat, np.array(tempEdgeMat)], axis=0) # 合并边的集合
        # print("triangleMat", triangleMat)
        # print("edgeMat", edgeMat.shape)
        # 删除一开始初始化的最大的三角形maxtriangle
        tempDel = []
        for k in range(triangleMat.shape[0]):
            if 0 in triangleMat[k, :] or 1 in triangleMat[k, :] or 2 in triangleMat[k, :] :
                tempDel.append(k)
        # print(tempDel)
        triangleMat = np.delete(triangleMat, tempDel, axis=0) # axis=0 表示删除行
        edgeMat = np.concatenate([triangleMat[:, [0, 1]], triangleMat[:, [1, 2]], triangleMat[:, [2, 0]]], axis=0)
        # edgeMat = np.array(sorted(edgeMat.tolist(), key=lambda x: (x[0], x[1])))
        edgeMat = np.sort(edgeMat, axis=1)
        edgeMat = self._unranked_unique(edgeMat)

        return triangleMat, edgeMat 


        # tempEdgeMat = np.empty(shape=[0, 6], dtype=np.int)
        # tempTriangleMat = np.empty(shape=[0, 3], dtype=np.int)

    def plot_figure(self, triangleMat, title="Delaulay Triangle", save_path="./figures/Delaunay_Triangle.png"):
        plt.figure(figsize=(8, 8), facecolor='w')
        for i in range(triangleMat.shape[0]):
            # for j in range(2)
                # plt.plot()
            # point1 = [self.dots[triangleMat[i, 0], 0], self.dots[triangleMat[i, 0], 0]]
            plt.plot([self.dots[triangleMat[i, 0], 0], self.dots[triangleMat[i, 1], 0]], 
                        [self.dots[triangleMat[i, 0], 1], self.dots[triangleMat[i, 1], 1]], "b-")
            plt.plot([self.dots[triangleMat[i, 0], 0], self.dots[triangleMat[i, 2], 0]], 
                        [self.dots[triangleMat[i, 0], 1], self.dots[triangleMat[i, 2], 1]], "b-")
            plt.plot([self.dots[triangleMat[i, 1], 0], self.dots[triangleMat[i, 2], 0]], 
                        [self.dots[triangleMat[i, 1], 1], self.dots[triangleMat[i, 2], 1]], "b-")

        plt.title(title)
        plt.savefig(os.path.join(self.based_path, save_path))
        plt.show()

    def _makeTempTriangle(self, tempEdgeMat, i):
        """
        将输入的边和目标点组成三角形，最终返回组成的三角形的下标
        输入参数:
            tempEdgeMat: 输入的边的集合 N * 6 分别表示组成该边两个点的下标 两个点对应的x y值
            i: 表示目标点在self.dots中的下标
        输出参数:
            tempTriangle: N * 3 3表示组成该三角形的顶点的在dots中的下标
        """
        pointNum = tempEdgeMat[:, [0, 1]] # 得到所有顶点的坐标
        # print("pointNum", pointNum)
        pointLine = pointNum[pointNum != i].astype(np.int) # 得到所有的不是该顶点的point

        N = len(pointLine) # 边的个数其实就是pointLine的个数
        points = self.dots[pointLine, :]
        complexPoints = [complex(point[0], point[1]) for point in points]
        # print(complexPoints)
        diffComplexPoints = [complexPoint - complex(self.dots[i, 0], self.dots[i, 1]) for complexPoint in complexPoints]
        # print(diffComplexPoints)
        angle = np.angle(diffComplexPoints, deg=False) # 返回每个复数的角度 然后对复数进行排序
        index = np.argsort(angle).tolist()
        index.append(index[0]) # 这里将最开始的下标添加进去方便遍历
        # print(index)
        tempTriangle = np.empty(shape=[0, 3], dtype=np.int)
        for k in range(N):
            tempTriangle = np.append(tempTriangle, np.array([pointLine[index[k]], pointLine[index[k+1]], i]).reshape(1, -1), axis=0)
        return tempTriangle    

    def __convertData(self, dot1, dot2, arr1, arr2, flag=True):
        """
        转化数据类型
        输入参数:
            dot1 dot2 表示两个数值
            arr1 arr2 表示两个数组
            flag: True 表示默认转为list 否则则是array
        输出参数:
            表示将dot1 dot2 arr1 arr2连接的list
        """
        if flag:
            return np.concatenate([np.array([dot1, dot2]), arr1, arr2]).tolist()
        else:
            return np.concatenate([np.array([dot1, dot2]), arr1, arr2])

    def _unranked_unique(self, nparray):
        """
        输出二维list的去重不排序的结果
        输出参数: 
            nparray: 数组
        输出参数:
            T: 不自动排序但是去重的数组
        """
        # print("nparray", nparray)
        T = np.empty(shape=[0, nparray.shape[1]], dtype=np.int)
        for i in nparray:
            mask = True
            for j in T:
                if (i == j).all():
                    mask = False
            if mask:
                T = np.append(T, i.reshape(1, -1), axis=0)
        return T
        
    def _makeEdge(self, dot1, dot2, dot3):
        """
        将dot1 dot2 dot3 这三个点构成三条边
        输入参数:
            dot1 dot2 dot3: 三个点在dots中的序号
        输出参数:
            edgeMat: 三个点构成的三条边的集合
        """
        edge1 = self._makeOneEdge(dot1, dot2)
        edge2 = self._makeOneEdge(dot1, dot3)
        edge3 = self._makeOneEdge(dot2, dot3)
        edgeMat = np.concatenate([edge1, edge2, edge3], axis=0)
        return edgeMat
        # print("edge1:", edge1)

    def _makeTempEdge(self, dot1, dot2, dot3, targetdot):
        """
        将dot1 dot2 dot3 这三个点和targetdot构成三条边
        输入参数:
            dot1 dot2 dot3: 三个点在dots中的序号
            targetdot: 目标点
        输出参数:
            edgeMat: 这三个点和targetdot构成三条边
        """

        edge1 = self._makeOneEdge(dot1, targetdot)
        edge2 = self._makeOneEdge(dot2, targetdot)
        edge3 = self._makeOneEdge(dot3, targetdot)
        edgeMat = np.concatenate([edge1, edge2, edge3], axis=0)
        # print("edgeMat", edgeMat)
        return edgeMat

    def _makeOneEdge(self, dot1, dot2):
        """
        利用dot1 和 dot2 生成一条边返回 辅助函数_makeEdged的使用
        输入参数：
            dot1 dot2: 两个点在dots中的位置
        输出参数:
            edge: 顶点+值的组合 输出为一行内容四个值
        
        """
        if self.dots[dot1, 0] < self.dots[dot2, 0]:
            edge = self.__convertData(dot1, dot2, self.dots[dot1, :], self.dots[dot2, :], flag=False)
        elif self.dots[dot1, 0] == self.dots[dot2, 0]:
            # 如果两个点的x相等的话
            if self.dots[dot1, 1] < self.dots[dot2, 1]:
                # 判断两个点的y的大小
                edge = self.__convertData(dot1, dot2, self.dots[dot1, :], self.dots[dot2, :], flag=False)
            else:
                edge = self.__convertData(dot2, dot1, self.dots[dot2, :], self.dots[dot1, :], flag=False)
        else:
            edge = self.__convertData(dot2, dot1, self.dots[dot2, :], self.dots[dot1, :], flag=False)

        return edge.reshape(1, -1)

    def _detectArray(self, inputs, dim=3):
        if inputs.size == 0:
            # print("+++"*40)
            return np.empty(shape=[0, dim], dtype=np.int)
        else:
            return inputs

    def _locationPoint(self, dot1, dot2, dot3, tempPoint):
        """
        判断点在三角形外接圆的那个部分
        输入参数: 
            dot1 dot2 dot2: 三角形的三个顶点
            tempPoint: 需要判断的点
        输出参数:
            mask: 该点相对元三角形外接圆的相对位置 
            0表示外侧 1表示内部 2表示右侧
        """
        # print(dot1, dot2, dot3, tempPoint)
        x0, y0, r = self._getCircle(dot1, dot2, dot3)
        x, y = tempPoint
        if x0 + r < x:
            # 如果点在三角形外接圆的右侧
            mask = 2
        elif math.sqrt((x - x0) ** 2 + (y - y0) ** 2) < r:
            # 如果点在三角形外接圆的内部
            mask = 0
        else:
            # 点在三角形外接圆的外部 跳过
            mask = 1
        return mask

    def _getCircle(self, dot1, dot2, dot3):

        """
        给定三角形的三个点 返回三角形的外接圆的x y 和半径
        输入参数：
            dot1 dot2 dot3 表示三角形的三个点
        输出参数：
            a b： 表示三角形外接圆的x y值
            r： 表示三角形的外接圆的半径
        """

        x1, y1 = dot1
        x2, y2 = dot2
        x3, y3 = dot3

        a=((y2-y1)*(y3*y3-y1*y1+x3*x3-x1*x1)-(y3-y1)*(y2*y2-y1*y1+x2*x2-x1*x1))/(2.0*((x3-x1)*(y2-y1)-(x2-x1)*(y3-y1)))
        b=((x2-x1)*(x3*x3-x1*x1+y3*y3-y1*y1)-(x3-x1)*(x2*x2-x1*x1+y2*y2-y1*y1))/(2.0*((y3-y1)*(x2-x1)-(y2-y1)*(x3-x1)))
        r=math.sqrt((x1-a)*(x1-a)+(y1-b)*(y1-b))

        return a, b, r


if __name__ == "__main__":
    x = [43, 20, 34, 18, 12, 32, 40, 4, 44, 30, 6, 47, 23, 13, 38, 48, 36, 46, 50, 37, 21, 7, 28, 25, 10]
    y = [3, 43, 47, 31, 30, 39, 9, 33, 49, 36, 21, 48, 14, 34, 41, 4, 1, 44, 18, 24, 20, 11, 27, 42, 13]
    # x = [0.814723686393179, 0.905791937075619, 0.126986816293506]
    # y = [0.913375856139019, 0.632359246225410, 0.0975404049994095]
    # x = [0.814723686393179,
    #     0.905791937075619,
    #     0.126986816293506,
    #     0.913375856139019]
    # y = [0.632359246225410,
    #     0.0975404049994095,
    #     0.278498218867048,
    #     0.546881519204984]

    voronoi = myVoronoi(x, y)
    # voronoi.createDelaulay()

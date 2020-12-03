% 基于AABB的碰撞检测
close all;clear;clc;


Adots = [[0, 0, 0]; [1, 0, 0]; [1, 1, 0]; [0, 1, 0]; [0, 0, 1]; [1, 0 ,1]; [1, 1, 1]; [0, 1, 1]];
Bdots = [[1, 1, 1]; [2, 1, 1]; [2, 2, 1]; [1, 2, 1]; [1, 1, 2]; [2, 1, 2]; [2, 2, 2]; [1, 2, 2]];
Cdots = [[3, 3, 2]; [5, 3, 2]; [4, 5, 2]; [4, 4, 4]];
% dotsList = cat(3, Adots, Bdots, Cdots)
dotsStr = ['A', 'B', 'C', 'A'];


[mindots1, maxdots1] = getMinMaxDots(Adots);
[mindots2, maxdots2] = getMinMaxDots(Bdots);
[mindots3, maxdots3] = getMinMaxDots(Cdots);

minDots = [mindots1; mindots2; mindots3];
maxDots = [maxdots1; maxdots2; maxdots3];
dic = [1, 2, 3, 1];

for i=1:size(dotsStr, 2) - 1
    k = dic(i+1);
    judgeCollision(minDots(i, :), maxDots(i, :), dotsStr(i),...
                                    minDots(k, :), maxDots(k, :), dotsStr(k));
end


function mask = judgeCollision(minDots1, maxDots1, str1, minDots2, maxDots2, str2)
% 输入两个三维几何题的坐标 判断两个三维几何题是否会发生碰撞

% 比较两个圆心之间的距离 
% 如果两个圆心之间的距离大于两个圆的半径总和说明没有发生碰撞
rangeList = [1, 2, 3, 1];
res = 0;
for i=1:size(rangeList, 2)-1
    res = res + judgeTwoDim(minDots1([rangeList(i), rangeList(i+1)]),...
                    maxDots1([rangeList(i), rangeList(i+1)]), minDots2([rangeList(i), rangeList(i+1)]),...
                    maxDots2([rangeList(i), rangeList(i+1)]));
end


% 只有三个的坐标轴都不发生碰撞 才表明两个三维几何体不会发生碰撞
if res == 3
    mask = 1;
else
    mask = 0;
end

% 输出判断
outputDescription(str1, str2, mask);

end

function flag = judgeTwoDim(minDot1, maxDot1, minDot2, maxDot2)
% 对应坐标轴上的多边形是否发生重叠 进而判断是否发生碰撞
% 输入参数：
%   minDot1 minDot2： 表示两个凸包在两个坐标轴上最小的点
%   maxDot1 maxDot2： 表示两个凸包在两个坐标轴上最大的点
% 输出参数：
%   flag：
%   1 表示没有发生碰撞
%   0 表示发生了碰撞

flag = 1; % 1 表示没有发生碰撞
if (maxDot1(1) >= minDot2(1)) && (maxDot2(1) >= minDot1(1)) &&...
                    (maxDot1(2) >= minDot2(2)) && (maxDot2(2) >= minDot1(2))
    flag = 0; % 如果满足碰撞的条件 也就是对应坐标轴上的投影发生重叠
else
    flag = 1;
end

end


function [mindots, maxdots] = getMinMaxDots(dots)
% 获取传入的多边形的最小顶点和最大顶点
% 输入参数： 
%   dots： 输入凸包的三维坐标

xmax = max(dots(:, 1)); xmin = min(dots(:, 1));
ymax = max(dots(:, 2)); ymin = min(dots(:, 2));
zmax = max(dots(:, 3)); zmin = min(dots(:, 3));

mindots = [xmin, ymin, zmin];
maxdots = [xmax, ymax, zmax];
end


function [] = outputDescription(str1, str2, mask)
% 输出说明
% 输入mask 输出两个物体是否产生碰撞

if mask
    sprintf('There is no collision between object %s and object %s', str1, str2)
else
    sprintf('There is a collision between object %s and object %s', str1, str2)
    
end

end


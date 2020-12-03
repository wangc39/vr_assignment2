% 基于包围球的碰撞检测
close all;clear;clc;


Adots = [[0, 0, 0]; [1, 0, 0]; [1, 1, 0]; [0, 1, 0]; [0, 0, 1]; [1, 0 ,1]; [1, 1, 1]; [0, 1, 1]];
Bdots = [[1, 1, 1]; [2, 1, 1]; [2, 2, 1]; [1, 2, 1]; [1, 1, 2]; [2, 1, 2]; [2, 2, 2]; [1, 2, 2]];
Cdots = [[3, 3, 2]; [5, 3, 2]; [4, 5, 2]; [4, 4, 4]];
% dotsList = cat(3, Adots, Bdots, Cdots)
dotsStr = ['A', 'B', 'C', 'A'];

[center1, r1] = getSphereCenter(Adots);
[center2, r2] = getSphereCenter(Bdots);
[center3, r3] = getSphereCenter(Cdots);

centerList = [center1; center2; center3];
rList = [r1; r2; r3];


for i=1:size(dotsStr, 2)-1
    judgeCollision(centerList(i), r1, dotsStr(i), centerList(i+1), r2, dotsStr(i+1));
end

function mask = judgeCollision(center1, r1, str1, center2, r2, str2)
% 输入两个三维几何题的坐标 判断两个三维几何题是否会发生碰撞
% 输入参数：
%   center1 center2： 两个包围球的球心坐标
%   r1 r2： 两个包围球的球心半径
%   str1 str2： 说明是那个几何体
% 输出参数：
%   mask：
%   1 表示没有发生碰撞
%   0 表示发生碰撞

% 比较两个圆心之间的距离 
% 如果两个圆心之间的距离大于两个圆的半径总和说明没有发生碰撞
if pdist2(center1, center2) > (r1 + r2)
    mask = 1;
else
    mask = 0;
end

% 输出判断
outputDescription(str1, str2, mask);

end


function [centerDot, r] = getSphereCenter(dots)
% 输入三维几何题的坐标 返回该三维几何体球心坐标和球的半径
% 输入参数： 
%   dots： 输入凸包的三维坐标
xmax = max(dots(:, 1)); xmin = min(dots(:, 1));
ymax = max(dots(:, 2)); ymin = min(dots(:, 2));
zmax = max(dots(:, 3)); zmin = min(dots(:, 3));

% 求解球心的坐标
% 球心的坐标就是最大点和最小点的中间
x = mean([xmax, xmin]); y = mean([ymax, ymin]); z = mean([zmax, zmin]); 
centerDot = [x, y, z];
r = pdist2([xmax, ymax, zmax], [xmin, ymin, zmin]) / 2; % 计算两个点之间的距离除以2

end


function [] = outputDescription(str1, str2, mask)
%% 输出说明
% 输入mask 输出两个物体是否产生碰撞

if mask
    sprintf('There is no collision between object %s and object %s', str1, str2)
else
    sprintf('There is a collision between object %s and object %s', str1, str2)
    
end

end

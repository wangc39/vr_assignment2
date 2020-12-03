close all; clc; clear;

Adots = [[0, 0, 0]; [1, 0, 0]; [1, 1, 0]; [0, 1, 0]; [0, 0, 1]; [1, 0 ,1]; [1, 1, 1]; [0, 1, 1]];
Bdots = [[1, 1, 1]; [2, 1, 1]; [2, 2, 1]; [1, 2, 1]; [1, 1, 2]; [2, 1, 2]; [2, 2, 2]; [1, 2, 2]];
Cdots = [[3, 3, 2]; [5, 3, 2]; [4, 5, 2]; [4, 4, 4]];

iterations = 10;
flag1 = GJK(Adots, Bdots, iterations);
flag2 = GJK(Bdots, Cdots, iterations);
flag3 = GJK(Cdots, Adots, iterations);

flag = [flag1; flag2; flag3];
dotsStr = ['A', 'B', 'C', 'A'];
for i=1:size(flag, 1)
    outputDescription(dotsStr(i), dotsStr(i+1), flag(i));
end


function flag = GJK(shape1, shape2, iterations)
% GJK 碰撞算法
% 只适用于两个凸包物体
%
% Input:
%   shape1:
%   必须是三维的，我们的算法是基于三维的
%   判断。因此，输入形状必须是 N* 3 维的。
%
%   shape2:
%   与shape1相同，它必须是由三维几何图形组成的一组点
% 
%   iterations:
%   该算法试图构造一个3-simplex的四面体。通过一定的迭代次数，来判断物体是否发生碰撞。
%   比较低的迭代次数，算法的运算时间也会减少。而且随着两个物体之间重叠层度的增加，
%   算法需要的迭代次数也在减少。所以需要我们在迭代次数中做出权衡。
% outputs:
%   flag: 
%   true: 物体发生碰撞
%   false: 物体没有发生碰撞

v = [0.8 0.5 1]; % 方向向量
% 先选择两个点
[a,b] = pickLine(v,shape2,shape1);

% 选择第三个点 从而构建三角形(单纯性)
[a, b, c, flag] = pickTriangle(a ,b, shape2, shape1, iterations);

% 选择第四个点 构建四面体
if (flag == 1)  % 如果找到三角形 就可以尝试开始构建四面体 如果没找到三角形就直接结束
    [a,b,c,d,flag] = pickTetrahedron(a,b,c,shape2,shape1,iterations);
end

end


function [a,b,c,d,flag] = pickTetrahedron(a,b,c,shape1,shape2,iterations)

% 我们已经成功构建了2D的三角单纯性
% 现在我们需要检查原点是否在三维单纯形的内部 

% flag初始化为0 表示没有构建四面体
flag = 0;

ab = b-a;
ac = c-a;

% 垂直于三角形的面
abc = cross(ab,ac);
ao = -a;

% 原点在三角形的上面
if dot(abc, ao) > 0
    d = c;
    c = b;
    b = a;
    
    v = abc;
    % 四面体的新点
    a = support(shape2,shape1,v); 
    
% 原点在三角形的下面    
else
    d = b;
    b = a;
    v = -abc;
    % 构建四面体的新点
    a = support(shape2,shape1,v); 
end

% 允许迭代最多iterations次数来尝试构建四面体
for i = 1:iterations 
    ab = b-a;
    ao = -a;
    ac = c-a;
    ad = d-a;
    
    % 检查面ABC，ABD和ACD
    % 垂直于三角形的面
    abc = cross(ab,ac);
    
    if dot(abc, ao) > 0 % 面ABC在三角形的上方
        % continue
    else
        acd = cross(ac,ad);% 垂直三角形的面
        
        if dot(acd, ao) > 0 % 面ACD在三角形的上方
            % 把这个变成新的底边三角形。
            b = c;
            c = d;
            ab = ac;
            ac = ad;            
            abc = acd;     
        elseif dot(acd, ao) < 0
            adb = cross(ad,ab);% 垂直三角形的面
            
            if dot(adb, ao) > 0 % 面ADB在三角形的上方
                % 把这个变成新的底边三角形。
                c = b;
                b = d;              
                ac = ab;
                ab = ad;
                abc = adb;           
            else
                flag = 1; 
                break; % 原点在四面体的内部 退出
            end
        end
    end
    
    if dot(abc, ao) > 0 
        d = c;
        c = b;
        b = a;    
        v = abc;
        a = support(shape2,shape1,v); % 构建四面体的新点
    else %below
        d = b;
        b = a;
        v = -abc;
        a = support(shape2,shape1,v); % 构建四面体的新点
    end
end

end

function [a, b, c, flag] = pickTriangle(a, b, shape2, shape1, iterations)

% flag = 0 表示没有建立三角形
flag = 0;

% 第一次尝试
ab = b-a;
ao = -a;

% v垂直于ab，指向原点的大致方向。
v = cross(cross(ab,ao),ab);

c = b;
b = a;
a = support(shape2,shape1,v);

for i = 1:iterations 
    
    ab = b-a;
    ao = -a;
    ac = c-a;
    
    % 垂直于三角形的面
    abc = cross(ab,ac);
    
    % 垂直于AB远离三角形
    abp = cross(ab,abc);
    % 垂直于AC远离三角形
    acp = cross(abc,ac);
    
    % 首先，确保我们的三角形“包含”二维投影中的原点
    % 原点是否在AB之上
    if dot(abp,ao) > 0
        c = b; % 丢弃最远的一点，在正确的方向得到最新的一个点
        b = a;
        v = abp;
        
        % 原点是否在AC之上
    elseif dot(acp, ao) > 0
        b = a;
        v = acp;
        
    else
        flag = 1;
        break; % 成功构建三角形
    end
    a = support(shape2,shape1,v);
end

end

function [a, b] = pickLine(v, shape2, shape1)
% 构造单纯性的第一条直线
% 分别在方向向量v和-v上选择最远的点
% 从而可以构造出单纯性的第一条线段
b = support(shape2, shape1, v);
a = support(shape2, shape1, -v);
end

function point = support(shape2, shape1, v)
% 得到 Minkowski 差
% 分别得到在给定 v 和 -v方向上最远的两个点
point1 = getFartherPoint(shape1, v);
point2 = getFartherPoint(shape2, -v);
point = point1 - point2;
end

function point = getFartherPoint(shape, v)
% 找到在给定方向上，该几何体投影的点，也就是最远的点
x = shape(:, 1);
y = shape(:, 2);
z = shape(:, 3);
% 这一步操作相当于该几何体上所有的点和原点组成的向量和方向向量v点乘
% 然后计算出点乘之后的最大值 就是该几何体上沿着该方向向量最远的点
dotted = x*v(1) + y*v(2) + z*v(3);
[~, maxIdx] = max(dotted);
% 获得该几何体在该方向向量上最远的点
point = [x(maxIdx), y(maxIdx), z(maxIdx)];
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
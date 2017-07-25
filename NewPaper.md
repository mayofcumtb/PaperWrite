# title:
## abstract

## introduction
motivation
Deep learning technech is a data-driven algorithm. With the model tasj become compexity, tranditional deep learning methods need more and more labels to finish training process.
基于深度学习的目标检测算法是一种数据量有着一定要求的算法。随着深度学习所需处理任务变得越来越复杂，获取大量的训练样本变得越来越难困难，并且对于一些目标检测算法来说，通过人工获取的标记信息依然有一定的误差。随着对深度学习算法研究的深入，近年来，相关学者们

- 动机1： 深度学习算法需要大量的标记样本，而且随着任务复杂度的增加，人工方式获取的样本其可靠性也在降低，难度也越来越大
- 动机2： 解决样本问题有几种思路，
-- 1. 通过平移旋转，简单扩充样本 lack文章
-- 2. 通过无监督学习方法或弱监督学习方法来降低对人工标记样本的需求 lack
-- 3. 利用合成数据来进行样本的扩充，降低样本获取的人工成本 lack gan 和 其他

- 合成样本文章（1）cvpr2017 best paper apple inc （2）render for cnn 
- 快递目标检测算法（1）yolov2 （2）ssd


## related work



```matlab
% crop according to truncationParam
function [I, left, right, top, bottom, px, py, bb_width, bb_height] = crop_gray(I, bgColor, truncationParam)

%calculate the bounding box from the source  
threshold = graythresh(I);
binary_image = im2bw(I, threshold);
img_rec = regionprops(binary_image, 'boundingbox');
bbox = img_rec.BoundingBox;
% disp(bbox);
x_vertex = bbox(1); y_vertex = bbox(2); bbox_width = bbox(3); bbox_height =bbox(4);

% get boundaries of object
[nr, nc] = size(I);
colsum = sum(I == bgColor, 1) ~= nr;
rowsum = sum(I == bgColor, 2) ~= nc;

left = find(colsum, 1, 'first');
% disp(left);
if left == 0
    left = 1;
end
right = find(colsum, 1, 'last');
if right == 0
    right = length(colsum);
end
top = find(rowsum, 1, 'first');
if top == 0
    top = 1;
end
bottom = find(rowsum, 1, 'last');
if bottom == 0
    bottom = length(rowsum);
end

width = right - left + 1;
height = bottom - top + 1;

% strecth
if truncationParam(1) < 0.5
    dx1 = width * truncationParam(1); % left [-0.0336267000000000,-0.0452406000000000,-0.0612320000000000,0.162760000000000]
else
    dx1 = width * (truncationParam(1)-0.5);
end
if truncationParam(2) < 0.5
    dx2 = width * truncationParam(2); % right -0.0452406000000000 
else
    dx2 = width * (truncationParam(2) - 0.5);
end
if truncationParam(3) < 0.4
    dy1 = height * truncationParam(3); % top -0.0612320000000000
else
    dy1 = height * (truncationParam(3) - 0.4);
end
if truncationParam(4) <0.4
    dy2 = height * truncationParam(4); % bottom 0.162760000000000
else
    dy2 = height * (truncationParam(4) - 0.4);
end


leftnew = max([1, left + dx1]);
leftnew = min([leftnew, nc]);
rightnew = max([1, right + dx2]);
rightnew = min([rightnew, nc]);
if leftnew > rightnew
    leftnew = left;
    rightnew = right;
end

topnew = max([1, top + dy1]);
topnew = min([topnew, nr]);
bottomnew = max([1, bottom + dy2]);
bottomnew = min([bottomnew, nr]);
if topnew > bottomnew
    topnew = top;
    bottomnew = bottom;
end



left = round(leftnew); right = round(rightnew);
top = round(topnew); bottom = round(bottomnew);
% disp([left right top bottom]);
%disp([x_center, y_center, bbox_width, bbox_height]);

%x_center = x_center - left;
%y_center = y_center - bottom;


% image_info = [left, right, top, bottom];
% bbox_info = [px, py, bb_width, bb_height];ss
% calculate rate




x_vertex = x_vertex - left;
y_vertex = y_vertex - top;

I = I(top:bottom, left:right, :);
show = 1;
if show
    figure(1);
    imshow(I);
% disp([px, py, bb_width, bb_height]);
    x_vertex = round(x_vertex); y_vertex = round(y_vertex);

    rectangle('position', [x_vertex, y_vertex, bbox_width+2, bbox_height+2], 'EdgeColor', 'r');
end
px = x_vertex + bbox_width+2/2;
py = y_vertex + bbox_height+2/2;

bb_width = bbox_width+2;
bb_height = bbox_height+2;
px = round(px);
py = round(py);
bb_width = round(bb_width);
bb_height = round(bb_height);
```

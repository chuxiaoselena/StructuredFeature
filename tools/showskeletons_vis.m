function h = showskeletons_vis(im, points, pa, msize)
if nargin < 4
  msize = 4;
end
p_no = numel(pa);

switch p_no
  case 26
    partcolor = {'g','g','y','r','r','r','r','y','y','y','m','m','m','m','y','b','b','b','b','y','y','y','c','c','c','c'};
  case 14
    partcolor = {'g','g','y','r','r','y','m','m','y','b','b','y','c','c'};
  case 18
    partcolor = {'g','g','y','r','r','r','r','y','y','y','y','b','b','b','b','y','y','y'};
  case 16
    partcolor = {'g','g','y','r','r','r','r','y','y','y','y','b','b','b','b','y','y','y'};
  otherwise
    error('showboxes: not supported');
end
h = imshow(im); hold on;
if ~isempty(points)
  x = points(:,1);
  y = points(:,2);
  v = points(:,3);
  for child = 1:p_no
    parent = pa(child);
    if parent == -1 %|| v(parent) == 0 || v(child) == 0 
      continue;
    end
    
    x1 = x(parent);
    y1 = y(parent);
    x2 = x(child);
    y2 = y(child);
%     fprintf('%d %d %d %d\n', x1, y1, x2, y2);
    if v(child) ~= -1
    plot(x2, y2, 'o', 'color', partcolor{child}, ...
      'MarkerSize',msize, 'MarkerFaceColor', partcolor{child});
    end
    
    if v(child)~= -1 && v(parent)~=-1
    line([x1 x2],[y1 y2],'color',partcolor{child},'linewidth',round(msize/2));
    end
  end
end
% drawnow; hold on;

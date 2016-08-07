function h = showskeletons_joints(im, joints, pa, msize)
if nargin < 4
    msize = 5;
end
p_no = numel(pa);

switch p_no
    case 10
        partcolor = {'g','g','y','r','r','y','y','m','m','y','b','b','b','b','y','y','y','c','c','c','c'};
    case 26
        partcolor = {'g','g','y','r','r','r','r','y','y','y','m','m','m','m','y','b','b','b','b','y','y','y','c','c','c','c'};
    case 51
        partcolor = {'g','g','g','y','y','r','r','r','r','r','r','r','r','y','y','y','y','y','y',...
            'm','m','m','m','m','m','m','m','y','y','b','b','b','b','b','b','b','b','y','y','y','y','y','y',...
            'c','c','c','c','c','c','c','c'};
    case 33
        partcolor = {'g','g','g','y','r','r','r','r','y','y','y','y','m','m','m','m','m','m','y','b','b','b','b','y','y','y','y','c','c','c','c','c','c'};
    case 39
        partcolor = {'g','g','g','y','y','r','r','r','r','r','r','y','y','y','y','m','m','m','m','m','m',...
            'y','y','b','b','b','b','b','b','y','y','y','y','c','c','c','c','c','c'};
    case 14
        partcolor = {'g','g','y','r','r','y','m','m','y','b','b','y','c','c'};
    case 18
        partcolor = {'g','g','y','r','r','r','r','y','y','y','y','b','b','b','b','y','y','y'};
    otherwise
        partcolor = {'g','g','g','y','y','r','r','r','r','r','r','y','y','y','y','m','m','m','m','m','m',...
            'y','y','b','b','b','b','b','b','y','y','y','y','c','c','c','c','c','c'};
end
h = imshow(im); hold on;
if ~isempty(joints)
    %   box = joints(:,1:4*p_no);
    %   xy = reshape(box,size(box,1),4,p_no);
    %   xy = permute(xy,[1 3 2]);
    for n = 1:size(joints,3)
        %     x1 = xy(n,:,1); y1 = xy(n,:,2); x2 = xy(n,:,3); y2 = xy(n,:,4);
        x = joints(:,1);
        y = joints(:,2);
        for child = 2:p_no
            x1 = x(pa(child));
            y1 = y(pa(child));
            x2 = x(child);
            y2 = y(child);
            %             if (child == 2)
            plot(x1, y1, 'o', 'color', partcolor{child}, ...
                'MarkerSize',msize, 'MarkerFaceColor', partcolor{child});
            %             end
            plot(x2, y2, 'o', 'color', partcolor{child}, ...
                'MarkerSize',msize, 'MarkerFaceColor', partcolor{child});
            line([x1 x2],[y1 y2],'color',partcolor{child},'linewidth',round(2*msize/5));
        end
    end
end
drawnow; hold off;

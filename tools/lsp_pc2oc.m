function joints = lsp_pc2oc(joints)
% Convert person centric annotations to observer centric annotations.
% Please cite:
% @inproceedings{eichner2013appearance,
%  title={Appearance Sharing for Collective Human Pose Estimation},
%  author={Eichner, Marcin and Ferrari, Vittorio},
%  booktitle={Asian Conference on Computer Vision (ACCV)},
%  year={2012},
% }
% check x coordinate of left/right shoulder/hip position and if required flip the entire
% limb so that right means right from the observer's point of view.
% input order:
% 1 Right ankle
% 2 Right knee
% 3 Right hip
% 4 Left hip
% 5 Left knee
% 6 Left ankle
% 7 Right wrist
% 8 Right elbow
% 9 Right shoulder
% 10 Left shoulder
% 11 Left elbow
% 12 Left wrist
% 13 Neck
% 14 Head top
% in person centric view
%
% output order: in observer's centric view
% 1    2    3    4    5    6    7   8    9    10   11   12  13   14
% lank lkne lhip rhip rkne rank lwr lelb lsho rsho relb rwr hbot htop
%

for i=1:size(joints,3)
  torso_bottom = sum(joints(1:2,3:4,i),2)/2;
  torso_top = sum(joints(1:2,9:10,i),2)/2;
  tvec = torso_bottom-torso_top;
  normangle = atan2(tvec(2),tvec(1));
  norm_conf = joints(1:2,:,i)-repmat(torso_top,1,size(joints,2));
  norm_conf = [cos(-normangle) -sin(-normangle); sin(-normangle) cos(-normangle)]*norm_conf;
  norm_conf = [cos(pi/2) -sin(pi/2); sin(pi/2) cos(pi/2)]*norm_conf;
  % stickmen is normalized so that torso points always down
  if (norm_conf(1,3) > norm_conf(1,4))
    %flip legs
    temp = joints(:,1:3,i);
    joints(:,1:3,i) = fliplr(joints(:,4:6,i));
    joints(:,4:6,i) = fliplr(temp);
  end
  if (norm_conf(1,9) > norm_conf(1,10)) % flip
    %flip arms
    temp = joints(:,7:9,i);
    joints(:,7:9,i) = fliplr(joints(:,10:12,i));
    joints(:,10:12,i) = fliplr(temp);
  end
end


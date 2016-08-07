function sticks = flic_joint2stick(joints)
% The canonical joint order:
% 1 Nose
% 2 Neck
% 3 Left shoulder (from observer's perspective)
% 4 Left elbow
% 5 Left wrist
% 6 Left hip
% 7 Right shoulder
% 8 Right elbow
% 9 Right wrist
% 10 Right hip
%
% The canonical part stick order:
% 1 Torso
% 2 Head
% 3 Left Upper Arm
% 4 Left Lower Arm
% 5 Right Upper Arm
% 6 Right Lower Arm

stick_no = 6;
sticks = zeros(4, stick_no);
sticks(:, 1) = [joints(2, :), (joints(6, :) + joints(10, :))/2];            % Torso
sticks(:, 2) = [joints(1, :), joints(2, :)];                                % Head
sticks(:, 3) = [joints(3, :), joints(4, :)];                                % Left Upper Arm
sticks(:, 4) = [joints(4, :), joints(5, :)];                                % Left Lower Arm
sticks(:, 5) = [joints(7, :), joints(8, :)];                                % Right Upper Arm
sticks(:, 6) = [joints(8, :), joints(9, :)];                                % Right Lower Arm

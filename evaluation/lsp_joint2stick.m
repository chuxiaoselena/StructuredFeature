function sticks = lsp_joint2stick(joints)
% The canonical joint order:
% 1 Head top
% 2 Neck
% 3 Left shoulder
% 4 Left elbow
% 5 Left wrist
% 6 Left hip
% 7 Left knee
% 8 Left ankle
% 9 Right shoulder
% 10 Right elbow
% 11 Right wrist
% 12 Right hip
% 13 Right knee
% 14 Right ankle
% 
% The canonical part stick order:
% 1 Head
% 2 Torso
% 3 Left Upper Arm
% 4 Left Lower Arm
% 5 Left Upper Leg
% 6 Left Lower Leg
% 7 Right Upper Arm
% 8 Right Lower Arm
% 9 Right Upper Leg
% 10 Right Lower Leg

stick_n = 10; % number of stick
sticks = zeros(4, stick_n);
sticks(:, 1) = [joints(1, :), joints(2,:)];                                          % Head 
sticks(:, 2) = [(joints(3, :) + joints(9, :))/2, (joints(6, :) + joints(12, :))/2];  % Torso
sticks(:, 3) = [joints(3, :) , joints(4, :)];                                        % Left U.arms
sticks(:, 4) = [joints(4, :), joints(5, :)];                                         % Left L.arms
sticks(:, 5) = [joints(6, :), joints(7, :)];                                         % Left U.legs
sticks(:, 6) = [joints(7, :), joints(8, :)];                                         % Left L.legs
sticks(:, 7) = [joints(9, :), joints(10, :)];                                        % Right U.arms
sticks(:, 8) = [joints(10, :), joints(11, :)];                                       % Right L.arms
sticks(:, 9) = [joints(12, :), joints(13, :)];                                       % Right U.legs
sticks(:, 10) = [joints(13, :), joints(14, :)];                                      % Right L.legs

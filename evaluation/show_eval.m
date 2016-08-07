function show_eval(gt, ests, conf, eval_methods)
assert(iscell(eval_methods));
% --- part sticks ---
symmetry_part_id = conf.symmetry_part_id;
show_part_ids = conf.show_part_ids;
part_name = conf.part_name;

% ---- joints ----
symmetry_joint_id = conf.symmetry_joint_id;
show_joint_ids = conf.show_joint_ids;
reference_joints_pair = conf.reference_joints_pair;
joint_name = conf.joint_name;

for kk = 1:numel(eval_methods)
    m = eval_methods{kk};
    switch m
        case 'strict_pcp'
            % stric PCP evaluation
            [pcp_detail,is_matched] = eval_strict_pcp(gt, ests);
            pcp = (pcp_detail + pcp_detail(symmetry_part_id))/2;
            pcp = pcp(show_part_ids);
            fprintf('------------ strict PCP Evaluation -------------\n')
            fprintf('Parts      '); fprintf('& %s ', part_name{:}); fprintf('\n');
            fprintf('strict PCP '); fprintf('& %.1f  ', pcp*100); fprintf('\n');
        case 'pdj'
            % PDJ evaluation
            range = 0:0.01:0.5;
            accs = eval_pdj(gt, ests, reference_joints_pair);
            accs = (accs + accs(:,symmetry_joint_id)) / 2;
            accs = accs(:, show_joint_ids);
            % print
            fprintf('-------------- PDJ Evaluation ---------------\n')
            fprintf('Joints    '); fprintf('& %s ', joint_name{:}); fprintf('\n');
            sample_pdj_thresholds = [0.1, 0.2];
            for ii = 1:length(sample_pdj_thresholds)
                t = sample_pdj_thresholds(ii);
                idx = find(range == t);
                fprintf('PDJ@%.2f  ', t); fprintf('& %.1f ', accs(idx,:)*100); fprintf('\n');
            end
            % plot
            line_width = 3;
            p_color = {'g','y','b','r','c','k','m'};
            figure; hold on; grid on;
            for ii = 1:numel(show_joint_ids)
                plot(range, accs(:, ii), p_color{mod(ii, numel(p_color))+1}, 'linewidth', line_width);
            end
            leg_str = cell(numel(show_joint_ids), 1);
            for ii = 1:numel(show_joint_ids)
                leg_str{ii} = sprintf('%s', joint_name{ii});
            end
            h_leg = legend(leg_str);
            set(h_leg, 'location', 'northwest', 'linewidth', 3);
            
            axis([range(1),range(end), 0, 1]);
            set(gca,'ytick', 0:0.1:1);
            set(gca, 'linewidth', 3);
            hold off;
        otherwise
            fprintf('%s has not been implemented\n', m);
    end
end


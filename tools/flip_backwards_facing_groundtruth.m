function examples = flip_backwards_facing_groundtruth(examples)
% This function takes examples whose left/right joint label semantics are
% flipped from usual and swaps the shoulder, elbow and wrist locations.
% Note that this makes backwards-facing people have "incorrect" annotation
% according to human anatomy, but oh-so-right when it comes to training and
% testing front/back-agnostic 2D human layout models.
% Please cite:
%  @inproceedings{modec13,
%    title={MODEC: Multimodal Decomposable Models for Human Pose Estimation},
%    author={Sapp, Benjamin and Taskar, Ben},
%    booktitle={In Proc. CVPR},
%    year={2013},
%  }

%% find where left joints are all on the "wrong" side of the right joints
C = cat(3,examples.coords);
leftinds = lookupPart('lsho','lelb','lwri','lhip');
rightinds = lookupPart('rsho','relb','rwri','rhip');
jlinds = lookupPart('lsho', 'lhip');
jrinds = lookupPart('rsho', 'rhip');
backwards = find(all(C(1,jlinds,:) < C(1,jrinds,:),2));
%% flip
% lhip = squeeze(C(:,lookupPart('lhip'),:));
% lsho = squeeze(C(:,lookupPart('lsho'),:));
% rsho = squeeze(C(:,lookupPart('rsho'),:));
for i=backwards(:)'
   examples(i).coords(:,[leftinds rightinds]) = examples(i).coords(:,[rightinds leftinds]);   
end

fprintf('flipped %d / %d examples\n',length(backwards), length(examples));

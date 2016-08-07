function [inds numParts] = lookupPart(varargin)
% Returns the part # in the annotation for a string representation

key = get_key();
invkey = fields(key);
assert(key.(invkey{10})==10);


if nargout==2
    numParts = max(cellfun(@(x)key.(x), fieldnames(rmfield(key,'KEYPOINT_FLIPMAP'))));
end

if nargin == 0, 
    inds = key; 
    return; 
end

if nargin==1 && iscell(varargin{1})
    names = varargin{1};
else
    names = varargin;
end
inds = nan(length(names),1);
for i=1:length(names)
    if isfield(key,names{i});
        inds(i) = key.(names{i});
    end
end

function key = get_key()
% key.L_Shoulder = 1;
% key.R_Shoulder = 4;
% key.L_Hip      = 7;
% key.R_Hip      = 10;
key.lsho = 1;
key.lelb = 2;
key.lwri = 3;
key.rsho = 4;
key.relb = 5;
key.rwri = 6;
key.lhip = 7;
key.lkne = 8;
key.lank = 9;
key.rhip = 10;
key.rkne = 11;
key.rank = 12;
key.leye = 13;
key.reye = 14;
key.lear = 15;
key.rear = 16;
key.nose = 17;
key.msho = 18;
key.mhip = 19;
key.mear = 20;
key.mtorso = 21;
key.mluarm = 22;
key.mruarm = 23;
key.mllarm = 24;
key.mrlarm = 25;
key.mluleg = 26;
key.mruleg = 27;
key.mllleg = 28;
key.mrlleg = 29;

%support for alternate, original labels
key.L_Shoulder = 1;
key.L_Elbow    = 2;
key.L_Wrist    = 3;
key.R_Shoulder = 4;
key.R_Elbow    = 5;
key.R_Wrist    = 6;
key.L_Hip      = 7;
key.L_Knee     = 8;
key.L_Ankle    = 9;
key.R_Hip      = 10;
key.R_Knee     = 11;
key.R_Ankle    = 12;

key.L_Eye      = 13;
key.R_Eye      = 14;
key.L_Ear      = 15;
key.R_Ear      = 16;
key.Nose       = 17;

key.M_Shoulder = 18;
key.M_Hip      = 19;
key.M_Ear      = 20;
key.M_Torso    = 21;
key.M_LUpperArm    = 22;
key.M_RUpperArm    = 23;
key.M_LLowerArm    = 24;
key.M_RLowerArm    = 25;
key.M_LUpperLeg    = 26;
key.M_RUpperLeg    = 27;
key.M_LLowerLeg    = 28;
key.M_RLowerLeg    = 29;



key.KEYPOINT_FLIPMAP = [
    key.L_Shoulder key.R_Shoulder
    key.L_Elbow    key.R_Elbow
    key.L_Wrist    key.R_Wrist
    key.R_Shoulder key.L_Shoulder
    key.R_Elbow    key.L_Elbow
    key.R_Wrist    key.L_Wrist
    key.L_Hip      key.R_Hip
    key.L_Knee     key.R_Knee
    key.L_Ankle    key.R_Ankle
    key.R_Hip      key.L_Hip
    key.R_Knee     key.L_Knee
    key.R_Ankle    key.L_Ankle
    key.L_Eye      key.R_Eye
    key.R_Eye      key.L_Eye
    key.L_Ear      key.R_Ear
    key.R_Ear      key.L_Ear
    key.M_LUpperArm key.M_RUpperArm
    key.M_RUpperArm key.M_LUpperArm
    key.M_LLowerArm key.M_RLowerArm
    key.M_RLowerArm key.M_LLowerArm
    key.M_LUpperLeg key.M_RUpperLeg
    key.M_RUpperLeg key.M_LUpperLeg
    key.M_LLowerLeg key.M_RLowerLeg
    key.M_RLowerLeg key.M_LLowerLeg
    key.Nose key.Nose
    ];
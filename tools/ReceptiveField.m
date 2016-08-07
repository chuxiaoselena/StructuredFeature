%-------- RF ------------
% %            1      2       3         4 
stride = {1,1, 2, 1,1,2, 1,1, 1,2, 1,1, 1,2, 1,1,1,2,1,1};
kenel =  {3,3, 2, 3,3,2, 3,3, 3,2, 3,3, 3,2, 3,3,3,2,7,1};
RF = kenel{end};
for i = length(stride):-1:2
   RF = (RF-1)*(stride{i-1})+kenel{i-1};
end
fprintf('%d\n',RF);

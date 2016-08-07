% par save
function parsave(fname,data, varargin)
if ~isempty(varargin)
  var_name = varargin{1};
else
  var_name=genvarname(inputname(2));
end
eval([var_name '=data;']);
try
  save(fname,var_name,'-append')
catch
  save(fname,var_name)
end

function [ X_tst, X_trn, maxCol ] = EliminateZEROS( X_tst, X_trn)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

 % eliminate zeros
    %X_tst=X_tst(:,any(X_tst));
    %X_trn=X_trn(:,any(X_trn));
    %{  %}
   idx_tst = X_tst~=0;
   NoZeros_X_tst = sum(idx_tst,1);
   indexColToDel = 0;
   for jj= 1:length(NoZeros_X_tst)
       if NoZeros_X_tst(jj) < 1  % at least 1 value > 0 per column
           indexColToDel = indexColToDel + 1;
           X_tst_ColToDel(indexColToDel) = jj;
       end
   end
   if indexColToDel > 0
       X_tst(:,X_tst_ColToDel)=[];
       X_trn(:,X_tst_ColToDel)=[];
   end
   
   
     
   idx_trn = X_trn~=0;
   NoZeros_X_trn = sum(idx_trn,1);
   indexColToDel = 0;
   for jj= 1:length(NoZeros_X_trn)
       if NoZeros_X_trn(jj) < 1         % at least 1 value > 0 per column
           indexColToDel = indexColToDel + 1;
           X_trn_ColToDel(indexColToDel) = jj;
       end
   end
    
  if indexColToDel > 0
       X_tst(:,X_trn_ColToDel)=[];
       X_trn(:,X_trn_ColToDel)=[];
   end
 
    [maxRow,maxColXTrn] = size(X_trn);
    [maxRow,maxColXTst] = size(X_tst);
    if  maxColXTrn < maxColXTst
       maxCol = maxColXTrn; 
    else
       maxCol = maxColXTst;
    end;

end


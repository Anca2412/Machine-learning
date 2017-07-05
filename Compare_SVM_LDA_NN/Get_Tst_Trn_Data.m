function [ Y_trn, Y_tst, Histogram ] = Get_Tst_Trn_Data( X_trn, Y_trn, X_tst, Y_tst, trnFeature_Set1, tstFeature_Set1 )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    count = 0;
    for i=1:10
        for j=1:15
            Histogram = hist(double(trnFeature_Set1{i,j})', 0:bin:255)'./length(trnFeature_Set1{i,j});
            % We concatenate individual row vector to form a 1280 dimension feature
            % descriptor. The Feature is a 1 by 6656 row vector. 
            count = count + 1;
            X_trn(count,:)= Histogram(:)';
            Y_trn(count)=i;
        end;
    end;
    
    count = 0;
    for i=1:10
        for j=1:15
            Histogram = hist(double(tstFeature_Set1{i,j})', 0:bin:255)'./length(tstFeature_Set1{i,j});
            % We concatenate individual row vector to form a 1280 dimension feature
            % descriptor. The Feature is a 1 by 6656 row vector. 
            count = count + 1;
            X_tst(count,:)= Histogram(:)';
            Y_tst(count)=i;
        end;
    end;

end


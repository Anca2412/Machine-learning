clear; close all; clc; 

load('Feature.mat');

% Compute the histogram of each row independently.
% For example assume trnFeature_Set1{1,1} is a matrix of 128 by 3155.
% After compute the histogram using 52 bins (0:10:255), we obtain a 
% normalised histgram, a matrix of 128 by 52.

 index        = 0;
 resultSVM    = zeros(3,1);
 resultLDA    = zeros(3,1);
 resultNeurNet = zeros(3,1);
 X_axis       = zeros(3,1);
for bin=10:2:11
    clear Histogram; 
    clear X_tst; clear X_trn; 
    clear Y_tst; clear Y_trn;
    clear X_tst_Predict; clear Y_tst_Predict;
    index = index + 1;
    X_axis(index) = bin;
    Y_trn = zeros(150,1);
    Y_tst = zeros(150,1);
    %[ Y_trn, Y_tst, Histogram ] = Get_Tst_Trn_Data( X_trn, Y_trn, X_tst, Y_tst, trnFeature_Set1, tstFeature_Set1 );
    
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
    
    [ X_tst, X_trn, maxCol ] = EliminateZEROS( X_tst, X_trn);
    [ resultSVM ] = GetSVM( X_trn, Y_trn, X_tst, Y_tst,index, resultSVM, maxCol );
    [ resultLDA ] = GetLDA( X_trn, Y_trn, X_tst, Y_tst, maxCol, index, resultLDA );
    [ resultNeurNet ] = GetNeurNet(  X_trn, Y_trn, X_tst, Y_tst, index, resultNeurNet );
end;

plot(X_axis,resultSVM, X_axis,resultLDA, X_axis,resultNeurNet);
axis([10 100 10 100]);
xlabel('Number of bins')
ylabel('Accuracy')
legend('SVM','LDA linear','NN')
hold on


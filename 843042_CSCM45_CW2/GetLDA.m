function [ resultLDA ] = GetLDA( X_trn, Y_trn, X_tst, Y_tst, maxCol, index, resultLDA )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
        LDAModel = fitcdiscr(X_trn(:,1:maxCol),Y_trn); % Fit a LDA model
        Y_tst_Predict = predict(LDAModel,X_tst(:,1:maxCol));   %2347 first zero col with bins of size 12
        % Check accuracy
        F=confusionmat(Y_tst, Y_tst_Predict);
        Diff = Y_tst-Y_tst_Predict;
        ind = find(Diff==0);
        Right = numel(ind);
        Wrong = numel(Y_tst) - Right;
        Accuracy = Right/(Right+Wrong);
    %disp(['Accuracy on testing set is: ' num2str(Accuracy*100) '%']);
        resultLDA(index) = Accuracy*100;
end


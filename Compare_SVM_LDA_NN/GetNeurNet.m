function [ resultNeurNet ] = GetNeurNet(  X_trn, Y_trn, X_tst, Y_tst, index, resultNeurNet )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
net = feedforwardnet(25);
% Train NNs on the training dataset
XX_trn = X_trn(:,1:50);
XX_tst = X_tst(:,1:50);

net = train(net,XX_trn',Y_trn');
% Visualise the structure of NNs

Y_tst_Predict = net(XX_tst');

Y_tst_Predict = round(Y_tst_Predict');
% Check accuracy
nCorrectPredictions = sum(Y_tst==Y_tst_Predict);
Accuracy = nCorrectPredictions/length(Y_tst);
resultNeurNet(index) = Accuracy*100;
end


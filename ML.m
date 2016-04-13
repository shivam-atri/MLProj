data = (readtable('MLProj/mldata.csv'));
trainingsize = floor(height(data) * 0.80);
testsize = 682 - trainingsize;
trainingdata = (datasample(data,trainingsize,'Replace',false));
[testdata,indexes] = setdiff(data,trainingdata);

trainbestlabels = trainingdata(:,3);
trainsecondbestlabels = trainingdata(:,18);
trainbasicfeatures = trainingdata(:,[8,9,10,11]);

testbestlabels = testdata(:,3);
testsecondbestlabels = testdata(:,18);
testbasicfeatures = testdata(:,[8,9,10,11]);

% KNN Algorithm
NumNeigh = 3;
mdl = fitcknn(trainbasicfeatures,trainbestlabels,'NumNeighbors',NumNeigh,'Distance','euclidean');

labels = predict(mdl,testbasicfeatures);
testbstlabels = table2cell(testbestlabels);
testsecondbstlabels = table2cell(testsecondbestlabels);
err = 0;
for i = 1:size(labels)
    res = size(setdiff(labels(i,:),testbstlabels(i,:)));
    res2 = size(setdiff(labels(i,:),testsecondbstlabels(i,:)));
    res = res(:,2) + res2(:,2);
    if(res == 2)
        err = err + 1;
    end
end

KNNaccuracy = (testsize - err)/testsize;

% SVM
Mdl = fitcecoc(trainbasicfeatures,trainbestlabels);
labels = predict(Mdl,testbasicfeatures);
err = 0;
for i = 1:size(labels)
    res = size(setdiff(labels(i,:),testbstlabels(i,:)));
    res2 = size(setdiff(labels(i,:),testsecondbstlabels(i,:)));
    res = res(:,2) + res2(:,2);
    if(res == 2)
        err = err + 1;
    end
end

SVMaccuracy = (testsize - err)/testsize;

% Binary Decision Tree
tree = fitctree(trainbasicfeatures,trainbestlabels,'Prune','on');
%view(tree,'Mode','graph');
labels = predict(tree,testbasicfeatures);
err = 0;
for i = 1:size(labels)
    res = size(setdiff(labels(i,:),testbstlabels(i,:)));
    res2 = size(setdiff(labels(i,:),testsecondbstlabels(i,:)));
    res = res(:,2) + res2(:,2);
    if(res == 2)
        err = err + 1;
    end
end

Treeaccuracy = (testsize - err)/testsize;


% AdaBoost Tree Ensemble
NumTrainCycles = 5;
t = templateTree('Surrogate','off');
Ensemble = fitensemble(trainbasicfeatures,trainbestlabels,'AdaBoostM2',NumTrainCycles,t);
labels = predict(Ensemble,testbasicfeatures);
err = 0;
for i = 1:size(labels)
    res = size(setdiff(labels(i,:),testbstlabels(i,:)));
    res2 = size(setdiff(labels(i,:),testsecondbstlabels(i,:)));
    res = res(:,2) + res2(:,2);
    if(res == 2)
        err = err + 1;
    end
end

AdaboostTreeEnsembleaccuracy = (testsize - err)/testsize;

% Subspace KNN Ensemble
NumTrainCycles = 5;
t = templateKNN('NumNeighbors',5,'Standardize',0);
Ensemble = fitensemble(trainbasicfeatures,trainbestlabels,'Subspace',NumTrainCycles,t);
labels = predict(Ensemble,testbasicfeatures);
err = 0;
for i = 1:size(labels)
    res = size(setdiff(labels(i,:),testbstlabels(i,:)));
    res2 = size(setdiff(labels(i,:),testsecondbstlabels(i,:)));
    res = res(:,2) + res2(:,2);
    if(res == 2)
        err = err + 1;
    end
end

SubspaceKNNEnsembleaccuracy = (testsize - err)/testsize;

% Bagging
NumTrainCycles = 5;
t = templateKNN('NumNeighbors',5,'Standardize',0);
Ensemble = fitensemble(trainbasicfeatures,trainbestlabels,'Bag',NumTrainCycles,t);
labels = predict(Ensemble,testbasicfeatures);
err = 0;
for i = 1:size(labels)
    res = size(setdiff(labels(i,:),testbstlabels(i,:)));
    res2 = size(setdiff(labels(i,:),testsecondbstlabels(i,:)));
    res = res(:,2) + res2(:,2);
    if(res == 2)
        err = err + 1;
    end
end

Baggingaccuracy = (testsize - err)/testsize;
% LSBoost
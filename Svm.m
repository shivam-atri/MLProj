data = (readtable('MLProj/mldata_correct.csv'));
trainingsize = floor(height(data) * 0.80);
testsize = 682 - trainingsize;
trainingdata = (datasample(data,trainingsize,'Replace',false));
[testdata,indexes] = setdiff(data,trainingdata);

trainbestlabels = trainingdata(:,3);
trainsecondbestlabels = trainingdata(:,18);
trainbasicfeatures = trainingdata(:,[5,6,7]);

testbestlabels = testdata(:,3);
testsecondbestlabels = testdata(:,18);
testbasicfeatures = testdata(:,[5,6,7]);

% SVM
Mdl = fitcecoc(trainbasicfeatures,trainbestlabels);
labels = predict(Mdl,testbasicfeatures);
testbstlabels = table2cell(testbestlabels);
        testsecondbstlabels = table2cell(testsecondbestlabels);
        besterr = 0;
        secondbesterr = 0;
        for i = 1:size(labels)
            res = size(setdiff(labels(i,:),testbstlabels(i,:)));
            res2 = size(setdiff(labels(i,:),testsecondbstlabels(i,:)));
            if(res(:,2) == 1)
                besterr = besterr + 1;
            end
            res = res(:,2) + res2(:,2);
            if(res == 2)
                secondbesterr = secondbesterr + 1;
            end
        end

bestaccuracy = (testsize - besterr)/testsize;
secondbestaccuracy = (testsize - secondbesterr)/testsize;

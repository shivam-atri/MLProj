data = (readtable('MLProj/mldata_correct.csv'));
numFold = 10;
div = 1 - numFold/100;
NumNeigh = [1 3 5 7  9 11 13 15];
index = 1;
BestTreeaccuracy = 0;
SecondTreeaccuracy = 0;
    for l = 1:numFold
        trainingsize = floor(height(data) * div);
        testsize = 682 - trainingsize;
        trainingdata = (datasample(data,trainingsize,'Replace',false));
        [testdata,indexes] = setdiff(data,trainingdata);

        trainbestlabels = trainingdata(:,3);
        trainsecondbestlabels = trainingdata(:,18);
        trainbasicfeatures = trainingdata(:,[5,6,7]);

        testbestlabels = testdata(:,3);
        testsecondbestlabels = testdata(:,18);
        testbasicfeatures = testdata(:,[5,6,7]);

        % Decision Tree Algorithm
        tree = fitctree(trainbasicfeatures,trainbestlabels,'Prune','on');
        labels = predict(tree,testbasicfeatures);
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

        BestTreeaccuracy = BestTreeaccuracy + (testsize - besterr)/testsize;
        SecondTreeaccuracy = SecondTreeaccuracy + (testsize - secondbesterr)/testsize;
    end
    BestAccuracy(index) = BestTreeaccuracy/numFold;
    SecondAccuracy(index) = SecondTreeaccuracy/numFold;
    BestTreeaccuracy = 0;
    SecondTreeaccuracy = 0;
    index = index + 1;
 view(tree,'Mode','graph');
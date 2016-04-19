data = (readtable('MLProj/mldata_correct.csv'));
numFold = 10;
div = 1 - numFold/100;
NumNeigh = [1 3 5 7  9 11 13 15];
index = 1;
BestKNNaccuracy = 0;
SecondKNNaccuracy = 0;
for k = 1:length(NumNeigh)
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

        % KNN Algorithm
        mdl = fitcknn(trainbasicfeatures,trainbestlabels,'NumNeighbors',NumNeigh(k),'Distance','euclidean');

        labels = predict(mdl,testbasicfeatures);
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

        BestKNNaccuracy = BestKNNaccuracy + (testsize - besterr)/testsize;
        SecondKNNaccuracy = SecondKNNaccuracy + (testsize - secondbesterr)/testsize;
    end
    BestAccuracy(index) = BestKNNaccuracy/numFold;
    SecondAccuracy(index) = SecondKNNaccuracy/numFold;
    BestKNNaccuracy = 0;
    SecondKNNaccuracy = 0;
    index = index + 1;
end
plot(NumNeigh,BestAccuracy,'g-o',NumNeigh,SecondAccuracy,'b-o');
xlabel('Value of K');
ylabel('Accuracy');
legend({'Best', 'Second Best'},'Location','NorthEast');
title('Classification Accuracy for Different Values of K');
data = (readtable('MLProj/mldata_correct.csv'));
numFold = 10;
div = 1 - numFold/100;
TrainCycles = [1 5 10 15 25 35 50 75 100];
index = 1;
BestAdaboostaccuracy = 0;
SecondAdaboostaccuracy = 0;
for k = 1:length(TrainCycles)
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

        % Adaboost Algorithm
       t = templateTree('Surrogate','off');
        Ensemble = fitensemble(trainbasicfeatures,trainbestlabels,'AdaBoostM2',TrainCycles(k),t);
        labels = predict(Ensemble,testbasicfeatures);
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

        BestAdaboostaccuracy = BestAdaboostaccuracy + (testsize - besterr)/testsize;
        SecondAdaboostaccuracy = SecondAdaboostaccuracy + (testsize - secondbesterr)/testsize;
    end
    BestAccuracy(index) = BestAdaboostaccuracy/numFold;
    SecondAccuracy(index) = SecondAdaboostaccuracy/numFold;
    BestAdaboostaccuracy = 0;
    SecondAdaboostaccuracy = 0;
    index = index + 1;
end
plot(TrainCycles,BestAccuracy,'g-o',TrainCycles,SecondAccuracy,'b-o');
xlabel('Number of train Cycles');
ylabel('Accuracy');
legend({'Best', 'Second Best'},'Location','NorthEast');
title('Classification Accuracy for Different Training Cycles');
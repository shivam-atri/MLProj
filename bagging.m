% Bagging
data = (readtable('MLProj/mldata_correct.csv'));
div = 0.80;
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
leaf = [1 5 10];
nTrees = 100;
rng(9876,'twister');
savedRng = rng; % save the current RNG settings
trainbestlabelsforbagging = table2cell(trainbestlabels);
color = 'bgr';
for ii = 1:length(leaf)
   % Reinitialize the random number generator, so that the
   % random samples are the same for each leaf size
   rng(savedRng);
   % Create a bagged decision tree for each leaf size and plot out-of-bag
   % error 'oobError'
   b = TreeBagger(nTrees,trainbasicfeatures,trainbestlabelsforbagging,'OOBPred','on','MinLeaf',leaf(ii));
   plot(1- b.oobError,color(ii));
   hold on;
end
xlabel('Number of grown trees');
ylabel('Out-of-bag classification accuracy');
legend({'1 leaf', '5 leaves', '10 leaves'},'Location','NorthEast');
title('Classification Error for Different Leaf Sizes');
hold off;

nTrees = 30;
leaf = 5;
b = TreeBagger(nTrees,trainbasicfeatures,trainbestlabelsforbagging,'OOBPred','on','MinLeaf',leaf);
label = predict(b,table2array(testbasicfeatures));
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
Bestbaggingaccuracy = (testsize - besterr)/testsize;
Secondbaggingaccuracy = (testsize - secondbesterr)/testsize;

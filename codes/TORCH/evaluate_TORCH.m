function evaluation_info=evaluate_TORCH(XTrain,YTrain,XTest,YTest,LTest,LTrain,param)    

    XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1)); 
    XTest = bsxfun(@minus, XTest, mean(XTest, 1));

    YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1)); 
    YTest = bsxfun(@minus, YTest, mean(YTest, 1));

    
    fprintf('caculating S12...\n');
    S12 = zeros(param.num_class1, size(LTrain,2) - param.num_class1);
    index1 = 0;
    index2 = 0;
    for i = 1:size(LTrain,1)
        for j = 1:size(LTrain,2)
            if(LTrain(i,j) == 1 && j <= param.num_class1)
                index1 = j;
            end
            if (LTrain(i,j) == 1&& j>param.num_class1)
                    index2 = j;
            end
        end
        S12(index1,index2 - param.num_class1) = 1;
    end
    S1 = zeros(size(LTrain,2) - param.num_class1,size(LTrain,2) - param.num_class1);
    for i = 1:size(S12,1)
        index = find(S12(i,:)==1);
        for j = 1:length(index)
            for k = 1:length(index)
                S1(index(j),index(k)) = 1;
            end
        end
    end
   S2 = LTrain(:, (param.num_class1+1):end)' *  LTrain(:, (param.num_class1+1):end); S2(S2>0) = 1;

    tic;
    fprintf('training...\n');

    [Wx, Wy, B] = train_TORCH(XTrain, YTrain, param, LTrain, S1, S2);
    
    fprintf('evaluating...\n');
    
    %% Training Time
    traintime=toc;
    evaluation_info.trainT=traintime;
    
    %% image as query to retrieve text database
    BxTest = compactbit(XTest*Wx' >= 0);
    ByTrain = compactbit(B' >= 0);
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain(:,param.num_class1+1:end), LTest(:,param.num_class1+1:end));

    %% text as query to retrieve image database
    ByTest = compactbit(YTest*Wy' >= 0);
    BxTrain = compactbit(B' >= 0);
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain(:,param.num_class1+1:end), LTest(:,param.num_class1+1:end));

end

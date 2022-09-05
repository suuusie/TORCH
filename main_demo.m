close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));


db = {'FashionVC'};
% db = {'Ssense'};
% db = {'Food'};

hashmethods = {'TORCH'}
loopnbits = [32]; 
% loopnbits = [16 32 64 128];


for dbi = 1 :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    result_URL = ['./results/'];

    if ~isdir(result_URL)
        mkdir(result_URL);
    end
    result_name = [result_URL datestr(now) '_final_' db_name '_result_test' '.mat'];
    
    
    if strcmp(db_name, 'FashionVC')
        load(['./datasets/',db_name,'/image_vgg.mat']);
        load(['./datasets/',db_name,'/label.mat']);
        load(['./datasets/',db_name,'/tag.mat']);

        R = randperm(size(Label,1))';
        Image_vgg = Image_vgg(R,:);
        Tag = Tag(R,:);
        Label = Label(R,:);
                
        TRAINING_SIZE = 16862;
        QUERY_SIZE = 3000;
        DATABASE_SIZE = 16862;
        
        num_class1 = 8;
        
        XTest = Image_vgg(1:QUERY_SIZE,:);
        YTest = Tag(1:QUERY_SIZE,:);
        LTest = Label(1:QUERY_SIZE,:);
        
        XTrain = Image_vgg(QUERY_SIZE + 1:QUERY_SIZE + TRAINING_SIZE,:);
        YTrain = Tag(QUERY_SIZE + 1:QUERY_SIZE + TRAINING_SIZE,:);
        LTrain = Label(QUERY_SIZE + 1:QUERY_SIZE + TRAINING_SIZE,:);
        
        XRetri = Image_vgg(QUERY_SIZE + 1:QUERY_SIZE + DATABASE_SIZE,:);
        YRetri = Tag(QUERY_SIZE + 1:QUERY_SIZE + DATABASE_SIZE,:);
        LRetri = Label(QUERY_SIZE + 1:QUERY_SIZE + DATABASE_SIZE,:);

        clear Image_vgg Label Tag
        
 elseif strcmp(db_name, 'Ssense')
        load(['./datasets/',db_name,'/image_vgg.mat']);
        load(['./datasets/',db_name,'/label.mat']);
        load(['./datasets/',db_name,'/tag.mat']);
        
        R = randperm(size(Label,1))';
        Image_vgg = Image_vgg(R,:);
        Tag = Tag(R,:);
        Label = Label(R,:);

        
        TRAINING_SIZE = 13696;
        QUERY_SIZE = 2000;
        DATABASE_SIZE = 13696;
        
        num_class1 = 4;

        
        XTest = Image_vgg(1:QUERY_SIZE,:);
        YTest = Tag(1:QUERY_SIZE,:);
        LTest = Label(1:QUERY_SIZE,:);
        
        XTrain = Image_vgg(QUERY_SIZE + 1:QUERY_SIZE + TRAINING_SIZE,:);
        YTrain = Tag(QUERY_SIZE + 1:QUERY_SIZE + TRAINING_SIZE,:);
        LTrain = Label(QUERY_SIZE + 1:QUERY_SIZE + TRAINING_SIZE,:);
        
        XRetri = Image_vgg(QUERY_SIZE + 1:QUERY_SIZE + DATABASE_SIZE,:);
        YRetri = Tag(QUERY_SIZE + 1:QUERY_SIZE + DATABASE_SIZE,:);
        LRetri = Label(QUERY_SIZE + 1:QUERY_SIZE + DATABASE_SIZE,:);

        clear Image_vgg Label Tag   
        
    end
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    
    
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        param.nbits = loopnbits(ii);
        param.num_class1 = num_class1;
        for jj = 1:length(hashmethods)
            
            switch(hashmethods{jj})
                    case 'TORCH'
                    fprintf('......%s start...... \n\n', 'TORCH');
                    param.eta1 = 10; param.eta2 = 0.2; param.eta3 = 1-param.eta2; param.eta4 = 10 ; param.eta5 = 10;
                    param.iter = 5; param.gamma = 1000;
                    param.rho = 1e-4;
                    OURparam = param;
                    eva_info_ = evaluate_TORCH(XTrain,YTrain,XTest,YTest,LTest,LTrain,OURparam);                      
            end
            eva_info{jj,ii} = eva_info_;
            clear eva_info_
        end
    end
    
    
    %% MAP
    for ii = 1:length(loopnbits)
        for jj = 1:length(hashmethods)
            Table_ItoT_MAP(jj,ii) = eva_info{jj,ii}.Image_VS_Text_MAP;
            Table_TtoI_MAP(jj,ii) = eva_info{jj,ii}.Text_VS_Image_MAP;
            
            trainT{ii}{jj} = eva_info{jj,ii}.trainT;

        end
    end

    
    %% Save
    save(result_name,'eva_info','param','loopnbits','hashmethods',...
       'Table_ItoT_MAP','Table_TtoI_MAP',...
       'trainT',...
        '-v7.3');
end

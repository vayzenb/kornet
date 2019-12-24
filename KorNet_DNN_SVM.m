clear all
rnet = resnet101;
vnet = vgg19;

%stimTypes = {'Outline', 'IC', 'Pert', 'Silh_White', 'Silh_Black', 'Pert_White', 'Pert_Black'};
stimTypes = {'Outline','Pert', 'IC'};
suf = {'','_ripple', '_IC'};
load('Stim/knLabels');

trainLabels = 1:length(knLabels);
resAct_train= zeros(length(knLabels), 1000);
resAct_test{1}= zeros(length(knLabels), 1000);

vggAct_train= zeros(length(knLabels), 1000);
vggAct_test{1}= zeros(length(knLabels), 1000);
for jj = 1:length(knLabels)
   %load training data
    IM = imread(['Stim/Final/Silh_Black/OBJ (', int2str(knLabels{jj,2}),').jpg']);
    IM = imresize(IM, [224,224]);
    
    vggAct_train(jj,:) = activations(vnet,IM,'fc8','OutputAs','rows');
    resAct_train(jj,:) = activations(rnet,IM,'fc1000','OutputAs','rows');
end

%Train SVM
SVMModel_vgg = fitcecoc(vggAct_train,trainLabels);
SVMModel_res = fitcecoc(resAct_train,trainLabels);

for ii = 1:length(stimTypes)
    for jj = 1:length(knLabels)
        IM = imread(['Stim/Final/', stimTypes{ii},'/OBJ (', int2str(knLabels{jj,2}), ')', suf{ii},'.png']);
        IM = imresize(IM, [224,224]);
        if strcmp(suf{ii},'_IC')
            IM = cat(3, IM, IM, IM);
        end

        vggAct_test{ii}(jj,:) = activations(vnet,IM,'fc8','OutputAs','rows');
        resAct_test{ii}(jj,:) = activations(rnet,IM,'fc1000','OutputAs','rows');
    end
    
    vggLabels = predict(SVMModel_vgg, vggAct_test{ii});
    resLabels = predict(SVMModel_res, resAct_test{ii});
    
    SVMAcc(ii,1) = mean(trainLabels == vggLabels');
    SVMAcc(ii,2) = mean(trainLabels == resLabels');
end


%% Do essentially RSA here
%Forced choice using pearson correlation
vgg_FC_corr = zeros(length(stimTypes), length(stimTypes));
res_FC_corr = zeros(length(stimTypes), length(stimTypes));
for ii = 1:length(stimTypes)
    for jj = 1:length(knLabels)
        target_corr = norm(vggAct_train(jj,:)- vggAct_test{ii}(jj,:));
        %target_corr =target_corr(1,2);
         for kk = 1:length(knLabels)
             currTrial = norm(vggAct_train(kk,:)- vggAct_test{ii}(jj,:));
         %    currTrial = currTrial(1,2);
          %   vgg_FC_corr(jj,kk) = currTrial;
             if target_corr <= currTrial
                 vgg_FC_corr(jj,kk) = 1;
             else
                 vgg_FC_corr(jj,kk) = 0;
             end
             
         end
        
    
    
    end
end

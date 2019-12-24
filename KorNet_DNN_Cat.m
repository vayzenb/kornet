clear all
rnet = resnet101;
vnet = vgg19;

%stimTypes = {'Outline', 'IC', 'Pert', 'Silh_White', 'Silh_Black', 'Pert_White', 'Pert_Black'};
stimTypes = {'Outline', 'Pert', 'IC', 'Silh_Black'};
suf = {'', '_ripple', '_IC', ''};
load('Stim/knLabels');

%IC needs to be concatenated

for ii = 1:length(stimTypes)
    %imFiles = [dir(['Stim/Final/', stimTypes{ii},'/', '*.png']); dir(['Stim/Final/', stimTypes{ii},'/', '*.jpg'])];
    
    for jj = 1:length(knLabels)
    
        if strcmp(stimTypes{ii}, 'Silh_Black')
            IM = imread(['Stim/Final/', stimTypes{ii},'/OBJ (', int2str(knLabels{jj,2}), ')', suf{ii},'.jpg']);
        else
            IM = imread(['Stim/Final/', stimTypes{ii},'/OBJ (', int2str(knLabels{jj,2}), ')', suf{ii},'.png']);
            
        end
        IM = imresize(IM, [224,224]);
        if strcmp(suf{ii},'_IC')
            IM = cat(3, IM, IM, IM);
        end

        %predict label from resnet
        rlabel = predict(rnet,IM);
        [~,p1] = sort(rlabel,'descend');


        %predict label from VGG
        vlabel = predict(vnet,IM);
        [~,p2] = sort(vlabel,'descend');

        for kk = 1:5
            ResLabels{ii}{jj, 1} = knLabels(jj);
            ResLabels{ii}{jj, kk+1} = cellstr(rnet.Layers(347,1).Classes(p1(kk)));
            vggLabels {ii}{jj, 1} = knLabels(jj);
            vggLabels {ii}{jj, kk+1} = cellstr(vnet.Layers(47,1).Classes(p2(kk)));
        end  
    
%     
%     ResLabels{jj,1} = imFiles(jj).name(1:(end-4));
%     ResLabels{jj,ii+1} = cellstr(rlabel);
%     
%     vggLabels{jj,1} = imFiles(jj).name(1:(end-4));
%     vggLabels{jj,ii+1} = cellstr(vlabel);
%     
    end
    
end



% label = predict(rnet,IM)'
%   [~,p] = sort(Data,'descend');
%    r = 1:length(Data);
%    r(p) = r;
%http://www.cs.virginia.edu/~vicente/entrylevel/data/translation-text-based-iccv2013-all.html
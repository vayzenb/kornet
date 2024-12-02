
clear all;
close all;
%% Computes area of convex hull and major/minor axis ratio
git_folder = 'C:\Users\Vlad\Desktop\git_repos/kornet/';

% read in stim names
stim_folder = [git_folder,'stim/test/Outline_Black_Filled/'];
stim_classes = readtable("stim\kornet_classes.csv");
num_items = size(stim_classes);

stim_classes.conv_hull = zeros(num_items(1),1);
stim_classes.axis_ratio = zeros(num_items(1),1);

for kk = 1:num_items(1)
    stim_num = stim_classes(kk, 2).(1);
    im = imread([stim_folder,'OBJ (',int2str(stim_num), ').png']);
    im = rgb2gray(im);

    BW = ~imbinarize(im);
    
    bw1= edge(BW);

    boundaries = bwboundaries(BW,'noholes');
    contour = boundaries{1};
    %compute convex hull area and ratio of major/minor axes
    [ch, cha] = convhull(contour);
    stats = regionprops(bw1, 'MajorAxisLength', 'MinorAxisLength');

    axis_ratio = stats(1).MajorAxisLength/stats(1).MinorAxisLength;
    
    stim_classes.conv_hull(kk) = round(cha,4);
    stim_classes.axis_ratio(kk) = axis_ratio;

end

%save
writetable(stim_classes,[git_folder,'results/stim_size.csv'],'Delimiter',',')


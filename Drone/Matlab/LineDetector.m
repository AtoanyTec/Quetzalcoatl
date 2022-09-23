clc
clear all
close all

I = imread('Recto_derecha.jpg');
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

BW = Binarizar_rojo(R,G,B);

imshow(BW)

[B,L] = bwboundaries(BW,'noholes');

% imshow(label2rgb(L, @jet, [.1 .1 .1]))

hold on

for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
end


s = regionprops(BW, 'Orientation');
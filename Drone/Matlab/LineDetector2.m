clc
clear all
close all

I = imread('Diagonal_izquierda.jpg');
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

BW = Binarizar_rojo(R,G,B);

[m, n] = size(BW);

figure
imshow(BW)
title('Segmentación de pista')

bordes = edge(BW,'canny');

bordes_martix = zeros(size(bordes));
bordesRGB = zeros([size(bordes),3]);
bordesRGB(:,:,1)= bordes*255;
bordesRGB(:,:,2)= bordes_martix;
bordesRGB(:,:,3)= bordes_martix;
figure
% imshow(bordesRGB)
imshow(bordes)
title('Lo que la computadora ve')

pista_RGB = zeros([size(BW),3]);
pista_RGB(:,:,1) = uint8(double(BW)*255);
pista_RGB(:,:,2) = uint8(double(BW)*255);
pista_RGB(:,:,3) = uint8(double(BW)*255);

figure
imshow(pista_RGB + bordesRGB)
title('Imagen de usuario')


pvertical = round(m/2);
phorizontal = round(n/2);

n_elementos = 20;
vector_horizontal = (phorizontal-(n_elementos/2)):1:(phorizontal+(n_elementos/2));
vector_vertical = ones(length(vector_horizontal))*pvertical;

hold on
plot(vector_horizontal, vector_vertical, 'g', 'LineWidth',2)


[Gmag, Gdir] = imgradient(BW,'prewitt');



% Gdir(vector_vertical(1), vector_horizontal(1):vector_horizontal(end))
yaw = Gdir(vector_vertical(1), 1:n);

indice = find(yaw~=0);

giro = yaw(indice(1)) + 180














